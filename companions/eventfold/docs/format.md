# EventFold record format v1

An EventFold stream carries an ordered sequence of opaque byte strings. Each record reconstructs exactly one input frame. Only the last decoded frame supplies history. Start every independent stream with an empty previous frame. A literal record can restart decoding at a known record boundary. The CLI and Python stream format prepend the four ASCII bytes `EFL1` (hex `45 46 4c 31`) once, then concatenate the records below. An empty EFL1 stream consists of just these four bytes. The C record API operates inside that envelope. There is no footer, checksum, or channel identifier. Applications must agree on framing/version and carry records in order on a reliable transport. Use an independent history for each logical stream.

| First byte | Record | Fields after the first byte |
|---|---|---|
| `00`–`3f` | Literal | Extended length, if needed, then the literal bytes |
| `40`–`7f` | Patch | Extended length, if needed, then prefix length, suffix length, and inserted bytes |
| `80` | Repeat | None |
| `81` | Two-splice | Six lengths, then two inserted byte strings |
| `82`–`ff` | Reserved | Reject |

In literal and patch records the low six bits give the number of literal or inserted bytes. When the low six bits are 63, the next unsigned LEB128 integer gives the number **beyond 63**. Thus exactly 63 literal bytes have header `3f 00`. Every integer, including prefix and suffix lengths, uses minimal unsigned LEB128, least significant group first. Reject overflow, an extra terminal zero group, reserved tags, or references outside the previous frame. C implementations use `size_t` for sizes. Receivers should apply their own negotiated resource limits before buffering data from a peer.

To apply a patch, emit the designated prefix of the previous frame, the inserted bytes, then the designated suffix of the previous frame. A zero-length suffix contributes nothing. Prefix and suffix must fit without overlap. Repeat emits the complete previous frame. Repeating the initial empty frame is valid.

A two-splice record encodes six unsigned LEB128 values in this order: prefix length, suffix length, old skip, middle length, left insertion length, right insertion length. The two literal insertions follow, left bytes then right bytes. Copy the prefix, left insertion, middle, right insertion, then suffix. The middle starts at offset `prefix + old_skip` in the previous frame. Require `prefix + old_skip + middle + suffix <= previous_length`. This permits arbitrary bytes in two separated regions to change while retaining a stable field between them.

The reference encoder first checks for exact equality. Otherwise it finds the longest common prefix followed by the longest common suffix that does not overlap that prefix in either frame. It chooses a patch only if its complete record uses fewer bytes than a literal. It then tries a two-splice record when both remaining interiors have at least 24 bytes and at least eight bytes survived in the prefix and suffix combined. Let `t = new_interior_length - 8`. For each eight-byte anchor at offset `t//2`, `t//4`, then `t - t//4`, search for its first occurrence in the old interior within 32 start positions either side of that offset. Extend this occurrence backwards and forwards within the two interiors for as long as the bytes match. Stop at the first span of at least 16 bytes. Use that candidate only if its record is strictly smaller than the current best. A decoder can accept other valid encodings of the same frame.

## Worked sequence

| Previous frame | Next frame | Encoded record in hex |
|---|---|---|
| empty | `abcdef` | `06 61 62 63 64 65 66` |
| `abcdef` | `abXdef` | `41 02 03 58` |
| `abXdef` | `abXdef` | `80` |
| `abXdef` | empty | `00` |
| empty | empty | `80` |

For a two-splice example, suppose the previous frame is `prefix:OLD--shared-envelope-field--abc:suffix` and the new frame is `prefix:NEW!--shared-envelope-field--xyz:suffix`. The reference encoder emits `81 07 07 03 19 04 03 4e 45 57 21 78 79 7a`. The 25-byte middle is copied directly from the previous frame.

## C interface

Include `eventfold.h`. Call `ef_bound` for the worst-case encoded size of one frame and `ef_encode` for the bytes written. Zero indicates failure in either function. The caller supplies separate output storage and owns the previous frame. The header performs no allocation or I/O.

`ef_decode` consumes exactly one record and accepts trailing bytes for later records. It reports `EF_MORE` for incomplete input. For a complete valid record it sets both the decoded length and the encoded bytes consumed, including when returning `EF_NOSPACE`. Retry with enough output storage before advancing the previous frame. At transport EOF a pending partial record is an error. Transport integrity matters because a corrupted literal or missing record can affect subsequent output.

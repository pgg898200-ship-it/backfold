"use strict";

const HEADER = Uint8Array.of(69, 70, 76, 49);
const INPUT_LIMIT = 512 * 1024;
const FRAME_LIMIT = 8192;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", {ignoreBOM: true});

function splitFrames(input) {
  if (input.length > INPUT_LIMIT) throw new Error("The demo accepts streams up to 512 KiB.");
  const frames = [];
  let frameStart = 0;
  let lineStart = 0;
  function finish(end) {
    if (frames.length === FRAME_LIMIT) throw new Error("The demo accepts up to 8,192 frames.");
    frames.push(input.subarray(frameStart, end));
    frameStart = end;
  }
  for (let i = 0; i < input.length;) {
    if (input[i] !== 10 && input[i] !== 13) {
      i++;
      continue;
    }
    const emptyLine = i === lineStart;
    i += input[i] === 13 && input[i + 1] === 10 ? 2 : 1;
    lineStart = i;
    if (emptyLine) finish(i);
  }
  if (frameStart !== input.length) finish(input.length);
  return frames;
}

function putVarint(bytes, n) {
  while (n >= 128) {
    bytes.push((n % 128) + 128);
    n = Math.floor(n / 128);
  }
  bytes.push(n);
}

function tagWithLength(op, n) {
  const bytes = [op + Math.min(63, n)];
  if (n >= 63) putVarint(bytes, n - 63);
  return Uint8Array.from(bytes);
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((sum, bytes) => sum + bytes.length, 0));
  let pos = 0;
  for (const bytes of parts) {
    out.set(bytes, pos);
    pos += bytes.length;
  }
  return out;
}

function findMiddle(old, current) {
  const last = current.length - 8;
  for (const anchor of [Math.floor(last / 2), Math.floor(last / 4), last - Math.floor(last / 4)]) {
    const hi = Math.min(old.length - 8, anchor + 32);
    for (let at = Math.max(0, anchor - 32); at <= hi; at++) {
      let run = 0;
      while (run < 8 && old[at + run] === current[anchor + run]) run++;
      if (run < 8) continue;
      let back = 0;
      while (back < Math.min(at, anchor) && old[at - back - 1] === current[anchor - back - 1]) back++;
      const limit = Math.min(old.length - at, current.length - anchor);
      while (run < limit && old[at + run] === current[anchor + run]) run++;
      if (back + run >= 16) return {skip: at - back, left: anchor - back, middle: run + back};
      break;
    }
  }
  return null;
}

function encodeRecord(prev, cur) {
  const limit = Math.min(prev.length, cur.length);
  let prefix = 0;
  while (prefix < limit && cur[prefix] === prev[prefix]) prefix++;
  if (prefix === cur.length && prefix === prev.length) return Uint8Array.of(128);
  let suffix = 0;
  while (suffix < limit - prefix && cur[cur.length - 1 - suffix] === prev[prev.length - 1 - suffix]) suffix++;
  const insert = cur.subarray(prefix, cur.length - suffix);
  const literalHeader = tagWithLength(0, cur.length);
  const patchHeader = tagWithLength(64, insert.length);
  const lengths = [];
  putVarint(lengths, prefix);
  putVarint(lengths, suffix);
  const patchCost = patchHeader.length + lengths.length + insert.length;
  const literalCost = literalHeader.length + cur.length;
  if (prefix + suffix >= 8 && insert.length >= 24 && prev.length - prefix - suffix >= 24) {
    const match = findMiddle(prev.subarray(prefix, prev.length - suffix), insert);
    if (match) {
      const {skip, left, middle} = match;
      const right = insert.length - left - middle;
      const header = [129];
      for (const field of [prefix, suffix, skip, middle, left, right]) putVarint(header, field);
      if (header.length + left + right < Math.min(literalCost, patchCost)) {
        return concat([header, insert.subarray(0, left), insert.subarray(left + middle)]);
      }
    }
  }
  return patchCost < literalCost
    ? concat([patchHeader, lengths, insert])
    : concat([literalHeader, cur]);
}

function pack(input) {
  const frames = splitFrames(input);
  let prev = new Uint8Array();
  const records = frames.map(frame => {
    const record = encodeRecord(prev, frame);
    prev = frame;
    return record;
  });
  return {frames, wire: concat([HEADER, ...records])};
}

function unpack(wire) {
  let pos = 0;
  function byte() {
    if (pos >= wire.length) throw new Error("Truncated EventFold stream.");
    return wire[pos++];
  }
  function varint() {
    let n = 0;
    for (let shift = 0; shift < 28; shift += 7) {
      const b = byte();
      n += (b & 127) * 2 ** shift;
      if (b < 128 && !(shift && !b) && n <= INPUT_LIMIT) return n;
      if (b < 128 || n > INPUT_LIMIT) break;
    }
    throw new Error("Invalid record length.");
  }
  function take(n) {
    if (n < 0 || n > wire.length - pos) throw new Error("Truncated EventFold record.");
    const data = wire.subarray(pos, pos + n);
    pos += n;
    return data;
  }
  for (const expected of HEADER) if (byte() !== expected) throw new Error("Missing EFL1 header.");
  const records = [];
  let prev = new Uint8Array();
  let total = 0;
  while (pos < wire.length) {
    const offset = pos;
    const tag = byte();
    let result, kind;
    if (tag === 128) {
      result = prev;
      kind = "Repeat";
    } else if (tag === 129) {
      const prefix = varint(), suffix = varint(), skip = varint();
      const middle = varint(), left = varint(), right = varint();
      if (prefix + skip + middle + suffix > prev.length) throw new Error("Invalid two-splice boundaries.");
      if (total + prefix + left + middle + right + suffix > INPUT_LIMIT) throw new Error("Decoded stream exceeds the demo limit.");
      result = concat([prev.subarray(0, prefix), take(left), prev.subarray(prefix + skip, prefix + skip + middle),
        take(right), prev.subarray(prev.length - suffix)]);
      kind = "Two-splice";
    } else if (tag <= 127) {
      let n = tag & 63;
      if (n === 63) n += varint();
      if (tag < 64) {
        result = take(n);
        kind = "Literal";
      } else {
        const prefix = varint(), suffix = varint();
        if (prefix + suffix > prev.length) throw new Error("Invalid patch boundaries.");
        if (total + prefix + n + suffix > INPUT_LIMIT) throw new Error("Decoded stream exceeds the demo limit.");
        result = concat([prev.subarray(0, prefix), take(n), prev.subarray(prev.length - suffix)]);
        kind = "Patch";
      }
    } else {
      throw new Error("Unknown EventFold record.");
    }
    total += result.length;
    if (total > INPUT_LIMIT || records.length >= FRAME_LIMIT) throw new Error("Decoded stream exceeds the demo limit.");
    records.push({offset, end: pos, kind, bytes: result});
    prev = result;
  }
  return records;
}

function sameBytes(a, b) {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

function samples() {
  function chat(delta, finish = null) {
    return "data: " + JSON.stringify({
      id: "chatcmpl-example", object: "chat.completion.chunk", created: 1750000000,
      model: "demo-model", choices: [{index: 0, delta, finish_reason: finish}]
    }) + "\n\n";
  }
  const content = ["Let", "’s", " check", " the", " next", " train", " to", " Malmö", "."];
  const call = (fragment, first = false) => chat({tool_calls: [{
    index: 0, ...(first ? {id: "call_1", type: "function"} : {}),
    function: {...(first ? {name: "train_departures"} : {}), arguments: fragment}
  }]});
  return {
    completion: chat({role: "assistant", content: ""}) +
      content.map(s => chat({content: s})).join("") + chat({}, "stop") + "data: [DONE]\n\n",
    tools: call("", true) + ['{"', "station", '":"', "Malmö", '",', '"limit"', ":3", "}"].map(s => call(s)).join("") +
      chat({}, "tool_calls") + "data: [DONE]\n\n",
    interleaved: ["started", "waiting", "ready", "running", "finished"].map((state, index) =>
      "data: " + JSON.stringify({seq: index, task: "station-lookup", worker: "schedule-service",
        route: "/stations/departures", status: state}) + "\n\n").join(""),
    mixed: ": heartbeat\r\n\r\n: heartbeat\r\n\r\nevent: update\ndata: first\n\nevent: update\ndata: second\n\n" +
      "id: 23\revent: end\rdata: café 🚆\r\rdata: tail without a terminator"
  };
}

function startInspector() {
  const el = id => document.getElementById(id);
  const examples = samples();
  const source = el("source"), sampleSelect = el("samples");
  let input = new Uint8Array(), records = [], wire = HEADER, chosen = 0, page = 0, revision = 0;
  const perPage = 7;
  const number = new Intl.NumberFormat("en-US");
  const count = n => number.format(n) + " B";
  function clearError() {
    el("error").hidden = true;
  }
  function reportError(message) {
    el("error").textContent = message;
    el("error").hidden = false;
    el("raw").textContent = el("packed").textContent = el("delta").textContent = "—";
    el("check").className = "";
    el("status").textContent = "Input unavailable";
    el("frame-count").textContent = "";
    records = [];
    chosen = page = 0;
    renderFrames();
  }
  function renderFrames() {
    const rows = el("rows");
    rows.replaceChildren();
    for (let i = page * perPage; i < Math.min((page + 1) * perPage, records.length); i++) {
      const record = records[i], row = document.createElement("tr");
      row.dataset.index = String(i);
      if (i === chosen) row.className = "chosen";
      const cell = row.insertCell(), pick = document.createElement("button");
      pick.className = "frame-pick";
      pick.textContent = "#" + (i + 1);
      pick.setAttribute("aria-label", "Inspect frame " + (i + 1));
      pick.setAttribute("aria-pressed", String(i === chosen));
      cell.append(pick);
      for (const val of [record.kind, count(record.bytes.length), count(record.end - record.offset)]) {
        row.insertCell().textContent = val;
      }
      row.addEventListener("click", () => {
        const keyboard = document.activeElement === pick;
        chosen = i;
        renderFrames();
        if (keyboard) rows.querySelector(".chosen button").focus();
      });
      rows.append(row);
    }
    el("range").textContent = records.length ? (page * perPage + 1) + "–" +
      Math.min(records.length, (page + 1) * perPage) + " of " + number.format(records.length) : "0 frames";
    el("prev").disabled = page === 0;
    el("next").disabled = (page + 1) * perPage >= records.length;
    const r = records[chosen];
    el("detail-title").textContent = r ? "Frame #" + (chosen + 1) + " · " + r.kind : "Frame detail";
    el("wire-offset").textContent = r ? "Wire offset " + number.format(r.offset) : "";
    const toHex = data => Array.from(data, n => n.toString(16).padStart(2, "0")).join(" ");
    const clip = (data, render) => render(data.subarray(0, 768)) + (data.length > 768 ? " …" : "");
    el("hex").textContent = r ? clip(wire.subarray(r.offset, r.end), toHex) : "No frame selected";
    el("view-format").textContent = "UTF-8";
    el("decoded").textContent = r ? clip(r.bytes, bytes => decoder.decode(bytes)) : "";
    if (r && !sameBytes(r.bytes, encoder.encode(decoder.decode(r.bytes)))) {
      el("view-format").textContent = "HEX";
      el("decoded").textContent = clip(r.bytes, toHex);
    }
  }
  function run(bytes) {
    clearError();
    input = bytes;
    try {
      const result = pack(bytes);
      wire = result.wire;
      records = unpack(wire);
      if (records.length !== result.frames.length || records.some((r, i) => !sameBytes(r.bytes, result.frames[i]))) {
        throw new Error("Byte verification failed.");
      }
      const growth = wire.length > bytes.length;
      el("raw").textContent = count(bytes.length);
      el("packed").textContent = count(wire.length);
      el("delta-label").textContent = growth ? "Size increase" : "Size reduction";
      el("delta").textContent = bytes.length ?
        (100 * Math.abs(bytes.length - wire.length) / bytes.length).toFixed(1) + "%" : "+4 B";
      el("delta").parentElement.classList.toggle("growth", growth);
      el("check").className = "good";
      el("status").textContent = "Exact round trip";
      el("frame-count").textContent = number.format(records.length) + " frame" + (records.length === 1 ? "" : "s") + " verified";
      chosen = Math.max(0, records.findIndex(r => r.kind !== "Literal"));
      page = Math.floor(chosen / perPage);
      renderFrames();
    } catch (err) {
      reportError(err.message);
    }
  }
  function setInput(bytes) {
    source.value = decoder.decode(bytes);
    run(bytes);
  }
  sampleSelect.onchange = () => {
    revision++;
    setInput(encoder.encode(examples[sampleSelect.value]));
  };
  source.oninput = () => {
    revision++;
    sampleSelect.value = "custom";
    run(encoder.encode(source.value));
  };
  source.onpaste = event => {
    if (!event.clipboardData) return;
    event.preventDefault();
    revision++;
    sampleSelect.value = "custom";
    const pasted = event.clipboardData.getData("text/plain");
    // A textarea normalizes CRLF; retain pasted CRLF and the original terminators around the selection.
    const original = decoder.decode(input);
    function sourceOffset(uiPos) {
      let real = 0;
      for (let seen = 0; seen < uiPos && real < original.length; seen++, real++) {
        if (original[real] === "\r" && original[real + 1] === "\n") real++;
      }
      return real;
    }
    const before = original.slice(0, sourceOffset(source.selectionStart));
    const after = original.slice(sourceOffset(source.selectionEnd));
    setInput(encoder.encode(before + pasted + after));
    const cursor = (before + pasted).replace(/\r\n?/g, "\n").length;
    source.setSelectionRange(cursor, cursor);
  };
  el("file").onchange = async event => {
    const file = event.target.files[0], currentRevision = ++revision;
    event.target.value = "";
    if (!file) return;
    sampleSelect.value = "custom";
    if (file.size > INPUT_LIMIT) {
      input = new Uint8Array();
      source.value = "";
      reportError("The demo accepts streams up to 512 KiB.");
      return;
    }
    try {
      const buffer = await file.arrayBuffer();
      if (currentRevision === revision) setInput(new Uint8Array(buffer));
    } catch {
      if (currentRevision === revision) reportError("The file could not be read.");
    }
  };
  for (const [id, direction] of [["prev", -1], ["next", 1]]) {
    el(id).onclick = () => {
      page += direction;
      chosen = page * perPage;
      renderFrames();
    };
  }
  setInput(encoder.encode(examples.completion));
}

if (typeof document !== "undefined") startInspector();
if (typeof module !== "undefined") module.exports = {splitFrames, encodeRecord, pack, unpack, samples};

// Separate translation unit prevents the compiler eliding writes to scratch.
#include <cstddef>
static volatile unsigned char last_byte;
extern "C" void consume_output(const unsigned char *data, std::size_t length) {
    if (length) last_byte = data[length - 1];
}

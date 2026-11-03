#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t le16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static double sqrt_approx(double value) {
    double guess;
    int i;
    if (value <= 0.0) return 0.0;
    guess = value > 1.0 ? value : 1.0;
    for (i = 0; i < 24; ++i) guess = 0.5 * (guess + value / guess);
    return guess;
}

static int fail(FILE *file, const char *message) {
    if (file) fclose(file);
    fprintf(stderr, "wav_probe: %s\n", message);
    return 2;
}

int main(int argc, char **argv) {
    FILE *file;
    unsigned char header[12], chunk[8], fmt[40], sample[4];
    long file_size, data_offset = -1;
    uint32_t riff_size, chunk_size, data_size = 0;
    uint16_t audio_format = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0, byte_rate = 0;
    uint64_t sample_count = 0;
    double sum_squares = 0.0, peak = 0.0, duration;

    if (argc != 2) {
        fprintf(stderr, "usage: wav_probe FILE.wav\n");
        return 64;
    }
    file = fopen(argv[1], "rb");
    if (!file) return fail(NULL, "cannot open input");
    if (fseek(file, 0, SEEK_END) != 0) return fail(file, "cannot seek input");
    file_size = ftell(file);
    if (file_size < 12 || fseek(file, 0, SEEK_SET) != 0) return fail(file, "truncated RIFF header");
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) return fail(file, "truncated RIFF header");
    if (memcmp(header, "RIFF", 4) || memcmp(header + 8, "WAVE", 4)) return fail(file, "not a RIFF/WAVE file");
    riff_size = le32(header + 4);
    if ((uint64_t)riff_size + 8U > (uint64_t)file_size) return fail(file, "truncated RIFF payload");

    while (ftell(file) >= 0 && ftell(file) + 8 <= file_size) {
        long payload;
        if (fread(chunk, 1, sizeof(chunk), file) != sizeof(chunk)) return fail(file, "truncated chunk header");
        chunk_size = le32(chunk + 4);
        payload = ftell(file);
        if ((uint64_t)payload + chunk_size > (uint64_t)file_size) return fail(file, "truncated chunk payload");
        if (!memcmp(chunk, "fmt ", 4)) {
            size_t wanted = chunk_size < sizeof(fmt) ? chunk_size : sizeof(fmt);
            if (chunk_size < 16 || fread(fmt, 1, wanted, file) != wanted) return fail(file, "invalid fmt chunk");
            audio_format = le16(fmt);
            channels = le16(fmt + 2);
            sample_rate = le32(fmt + 4);
            byte_rate = le32(fmt + 8);
            bits = le16(fmt + 14);
        } else if (!memcmp(chunk, "data", 4)) {
            data_offset = payload;
            data_size = chunk_size;
        }
        if (fseek(file, payload + chunk_size + (chunk_size & 1U), SEEK_SET) != 0) return fail(file, "invalid chunk boundary");
    }

    if (!audio_format || data_offset < 0) return fail(file, "missing fmt or data chunk");
    if (audio_format != 1) return fail(file, "only PCM WAV is supported");
    if (!channels || !sample_rate || !byte_rate || (bits != 16 && bits != 32)) return fail(file, "unsupported PCM layout");
    if ((uint64_t)data_offset + data_size > (uint64_t)file_size) return fail(file, "truncated data chunk");
    if (fseek(file, data_offset, SEEK_SET) != 0) return fail(file, "cannot seek to samples");

    while ((bits == 16 && fread(sample, 1, 2, file) == 2) ||
           (bits == 32 && fread(sample, 1, 4, file) == 4)) {
        double normalized;
        if (bits == 16) {
            int16_t value = (int16_t)le16(sample);
            normalized = (double)value / 32768.0;
        } else {
            int32_t value = (int32_t)le32(sample);
            normalized = (double)value / 2147483648.0;
        }
        if (normalized < 0.0) {
            if (-normalized > peak) peak = -normalized;
        } else if (normalized > peak) peak = normalized;
        sum_squares += normalized * normalized;
        ++sample_count;
        if (sample_count * (bits / 8U) >= data_size) break;
    }
    fclose(file);
    if (sample_count * (bits / 8U) != data_size) return fail(NULL, "truncated sample data");
    duration = (double)data_size / byte_rate;
    printf("CHANNELS=%u\nSAMPLE_RATE=%u\nBITS_PER_SAMPLE=%u\n", channels, sample_rate, bits);
    printf("DURATION_SECONDS=%.3f\nRMS=%.8f\nPEAK=%.8f\n", duration,
           sample_count ? sqrt_approx(sum_squares / sample_count) : 0.0, peak);
    if (peak == 0.0) {
        printf("SIGNAL=silent\n");
        return 3;
    }
    printf("SIGNAL=present\n");
    return 0;
}

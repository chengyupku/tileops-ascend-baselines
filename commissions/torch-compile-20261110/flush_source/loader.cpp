// Host ABI adapter for the unchanged exported device kernel. No device code.
// Same public ACL calls/launch grid/UBUF as TileLang ascend_module.cc.
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
static aclrtBinHandle binary = nullptr;
static aclrtFuncHandle function = nullptr;
static std::vector<unsigned char> data;
extern "C" int init(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return -1;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    if (size <= 0) { fclose(file); return -2; }
    data.resize(size);
    bool ok = fread(data.data(), 1, size, file) == static_cast<size_t>(size);
    fclose(file);
    if (!ok) return -3;
    int rc = aclrtBinaryLoadFromData(data.data(), data.size(), nullptr, &binary);
    if (rc) return rc;
    return aclrtBinaryGetFunction(binary, "_tilelang_profiler_cache_flush_kernel", &function);
}
extern "C" int call(void* buffer, void* stream) {
    aclrtLaunchKernelAttr attribute{};
    attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attribute.value.dynUBufSize = 32768;
    aclrtLaunchKernelCfg config{};
    config.attrs = &attribute;
    config.numAttrs = 1;
    return aclrtLaunchKernelWithHostArgs(function, 64, stream, &config,
                                        &buffer, sizeof(buffer), nullptr, 0);
}
extern "C" const char* get_last_error() { return aclGetRecentErrMsg(); }

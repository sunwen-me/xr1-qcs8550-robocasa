#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "QnnInterface.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpMem.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpProfile.h"
#include "System/QnnSystemInterface.h"

class DmaAllocation;

class ByteBuffer {
 public:
  ByteBuffer() = default;
  explicit ByteBuffer(size_t size) : owner_(std::make_shared<std::vector<uint8_t>>(size)) {
    auto *storage = static_cast<std::vector<uint8_t> *>(owner_.get());
    data_ = storage->data();
    size_ = storage->size();
  }

  static ByteBuffer from_payload(const std::shared_ptr<std::string> &payload, size_t offset,
                                 size_t size) {
    if (!payload || offset > payload->size() || size > payload->size() - offset) {
      throw std::runtime_error("invalid ByteBuffer payload view");
    }
    ByteBuffer result;
    result.owner_ = payload;
    result.data_ = reinterpret_cast<uint8_t *>(payload->data()) + offset;
    result.size_ = size;
    return result;
  }

  static ByteBuffer view(const ByteBuffer &source, size_t offset, size_t size) {
    if (offset > source.size() || size > source.size() - offset) {
      throw std::runtime_error("invalid ByteBuffer view");
    }
    ByteBuffer result;
    result.owner_ = source.owner_;
    result.dma_allocation_ = source.dma_allocation_;
    result.dma_offset_ = source.dma_offset_ + offset;
    result.data_ = const_cast<uint8_t *>(source.data()) + offset;
    result.size_ = size;
    return result;
  }

  static ByteBuffer dma_view(const std::shared_ptr<DmaAllocation> &allocation, size_t offset,
                             size_t size);

  uint8_t *data() { return data_; }
  const uint8_t *data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size() == 0; }
  const std::shared_ptr<DmaAllocation> &dma_allocation() const { return dma_allocation_; }
  size_t dma_offset() const { return dma_offset_; }

 private:
  std::shared_ptr<void> owner_;
  std::shared_ptr<DmaAllocation> dma_allocation_;
  size_t dma_offset_ = 0;
  uint8_t *data_ = nullptr;
  size_t size_ = 0;
};

class DmaAllocation {
 public:
  using RpcMemAllocFn = void *(*)(int, uint32_t, int);
  using RpcMemFreeFn = void (*)(void *);
  using RpcMemToFdFn = int (*)(void *);

  static std::shared_ptr<DmaAllocation> allocate(size_t size) {
    if (size == 0 || size > static_cast<size_t>(INT32_MAX)) {
      throw std::runtime_error("invalid shared DMA allocation size");
    }
    void *library = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      throw std::runtime_error("dlopen libcdsprpc.so failed: " + std::string(dlerror()));
    }
    const auto alloc = reinterpret_cast<RpcMemAllocFn>(dlsym(library, "rpcmem_alloc"));
    const auto free = reinterpret_cast<RpcMemFreeFn>(dlsym(library, "rpcmem_free"));
    const auto to_fd = reinterpret_cast<RpcMemToFdFn>(dlsym(library, "rpcmem_to_fd"));
    if (!alloc || !free || !to_fd) {
      dlclose(library);
      throw std::runtime_error("libcdsprpc.so has incomplete rpcmem API");
    }
    // Heap 25 is the system heap expected by QAIRT's HTP shared-buffer path.
    void *data = alloc(25, 1, static_cast<int>(size));
    if (!data) {
      dlclose(library);
      throw std::runtime_error("rpcmem_alloc failed for shared DMA buffer");
    }
    const int fd = to_fd(data);
    if (fd < 0) {
      free(data);
      dlclose(library);
      throw std::runtime_error("rpcmem_to_fd failed for shared DMA buffer");
    }
    return std::shared_ptr<DmaAllocation>(new DmaAllocation(library, free, fd, data, size));
  }

  ~DmaAllocation() {
    if (data_) {
      free_(data_);
    }
    if (library_) {
      dlclose(library_);
    }
  }

  int fd() const { return fd_; }
  uint8_t *data() const { return static_cast<uint8_t *>(data_); }
  size_t size() const { return size_; }

 private:
  DmaAllocation(void *library, RpcMemFreeFn free, int fd, void *data, size_t size)
      : library_(library), free_(free), fd_(fd), data_(data), size_(size) {}

  void *library_ = nullptr;
  RpcMemFreeFn free_ = nullptr;
  int fd_ = -1;
  void *data_ = nullptr;
  size_t size_ = 0;
};

ByteBuffer ByteBuffer::dma_view(const std::shared_ptr<DmaAllocation> &allocation, size_t offset,
                                size_t size) {
  if (!allocation || offset > allocation->size() || size > allocation->size() - offset) {
    throw std::runtime_error("invalid shared DMA buffer view");
  }
  ByteBuffer result;
  result.owner_ = allocation;
  result.dma_allocation_ = allocation;
  result.dma_offset_ = offset;
  result.data_ = allocation->data() + offset;
  result.size_ = size;
  return result;
}
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
using GetSystemProvidersFn =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *);

static constexpr uint32_t kMaxFrameBytes = 512u * 1024u * 1024u;
static constexpr uint32_t kEulerSteps = 5;
static constexpr size_t kActionBytes = 1u * 30u * 60u * sizeof(float);
static constexpr size_t kDmaAlignment = 4096;

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

static uint32_t read_u32_be(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24) |
         (static_cast<uint32_t>(data[1]) << 16) |
         (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

static uint64_t read_u64_be(const uint8_t *data) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value = (value << 8) | data[i];
  }
  return value;
}

static void append_u32_be(std::string *out, uint32_t value) {
  out->push_back(static_cast<char>((value >> 24) & 0xff));
  out->push_back(static_cast<char>((value >> 16) & 0xff));
  out->push_back(static_cast<char>((value >> 8) & 0xff));
  out->push_back(static_cast<char>(value & 0xff));
}

static bool recv_all(int fd, void *buffer, size_t size) {
  auto *cursor = static_cast<uint8_t *>(buffer);
  while (size != 0) {
    const ssize_t received = recv(fd, cursor, size, MSG_WAITALL);
    if (received <= 0) {
      return false;
    }
    cursor += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

static bool send_all(int fd, const void *buffer, size_t size) {
  const auto *cursor = static_cast<const uint8_t *>(buffer);
  while (size != 0) {
    const ssize_t sent = send(fd, cursor, size, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) {
      return false;
    }
    cursor += sent;
    size -= static_cast<size_t>(sent);
  }
  return true;
}

static volatile std::sig_atomic_t g_lifecycle_request = 0;

static void handle_enter_vla_signal(int) { g_lifecycle_request = 1; }
static void handle_exit_vla_signal(int) { g_lifecycle_request = 2; }

static uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count());
}

static size_t dtype_bytes(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    default:
      return 0;
  }
}

static uint64_t tensor_elements(const Qnn_Tensor_t &tensor) {
  const uint32_t rank = tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.rank : tensor.v1.rank;
  const uint32_t *dims = tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.dimensions
                                                                 : tensor.v1.dimensions;
  uint64_t elements = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    elements *= dims[i];
  }
  return elements;
}

static const char *tensor_name(const Qnn_Tensor_t &tensor) {
  return tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.name : tensor.v1.name;
}

static Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t &tensor) {
  return tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.dataType : tensor.v1.dataType;
}

static size_t tensor_bytes(const Qnn_Tensor_t &tensor) {
  const size_t element_size = dtype_bytes(tensor_dtype(tensor));
  if (element_size == 0) {
    throw std::runtime_error("unsupported QNN tensor dtype");
  }
  return static_cast<size_t>(tensor_elements(tensor)) * element_size;
}

static void bind_tensor(Qnn_Tensor_t *tensor, Qnn_TensorType_t type, void *data, size_t size) {
  if (tensor->version == QNN_TENSOR_VERSION_2) {
    tensor->v2.type = type;
    tensor->v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v2.clientBuf.data = data;
    tensor->v2.clientBuf.dataSize = static_cast<uint32_t>(size);
  } else {
    tensor->v1.type = type;
    tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v1.clientBuf.data = data;
    tensor->v1.clientBuf.dataSize = static_cast<uint32_t>(size);
  }
}

static void bind_tensor_memhandle(Qnn_Tensor_t *tensor, Qnn_TensorType_t type,
                                  Qnn_MemHandle_t handle) {
  if (tensor->version == QNN_TENSOR_VERSION_2) {
    tensor->v2.type = type;
    tensor->v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor->v2.memHandle = handle;
  } else {
    tensor->v1.type = type;
    tensor->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor->v1.memHandle = handle;
  }
}

static std::string read_text(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    return {};
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

static double read_double_file(const std::string &path) {
  const std::string text = read_text(path);
  if (text.empty()) {
    return -1.0;
  }
  char *end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  return end == text.c_str() ? -1.0 : value;
}

template <typename T>
static T parse_decimal(const char *text, T fallback) {
  if (!text) {
    return fallback;
  }
  T value = 0;
  const char *end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, value, 10);
  return result.ec == std::errc() && result.ptr != text ? value : fallback;
}

static uint32_t parse_profile_level(const char *text) {
  if (!text || std::strcmp(text, "basic") == 0) {
    return QNN_PROFILE_LEVEL_BASIC;
  }
  if (std::strcmp(text, "detailed") == 0) {
    return QNN_PROFILE_LEVEL_DETAILED;
  }
  if (std::strcmp(text, "linting") == 0) {
    return QNN_HTP_PROFILE_LEVEL_LINTING;
  }
  throw std::runtime_error("XR1_PROFILE_LEVEL must be basic, detailed, or linting");
}

static uint64_t process_cpu_ticks() {
  std::ifstream input("/proc/self/stat");
  std::string field;
  for (int i = 0; i < 14 && input >> field; ++i) {
    if (i == 13) {
      return parse_decimal<uint64_t>(field.c_str(), 0);
    }
  }
  return 0;
}

struct SystemCpuTicks {
  uint64_t busy = 0;
  uint64_t total = 0;

  SystemCpuTicks operator-(const SystemCpuTicks &other) const {
    SystemCpuTicks delta;
    delta.busy = busy >= other.busy ? busy - other.busy : 0;
    delta.total = total >= other.total ? total - other.total : 0;
    return delta;
  }
};

// Aggregate jiffies across all CPUs.  `busy` deliberately excludes idle and
// iowait so that busy/total is a real utilization ratio.  The previous version
// summed every field including idle, which made the derived percentage a
// constant ~100% regardless of load -- it was measuring wall time, not CPU.
static SystemCpuTicks system_cpu_ticks() {
  std::ifstream input("/proc/stat");
  std::string cpu;
  uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0,
           steal = 0;
  if (!(input >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal)) {
    return SystemCpuTicks{};
  }
  SystemCpuTicks ticks;
  ticks.busy = user + nice + system + irq + softirq + steal;
  ticks.total = ticks.busy + idle + iowait;
  return ticks;
}

static uint64_t rss_kb() {
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream fields(line.substr(6));
      uint64_t value = 0;
      fields >> value;
      return value;
    }
  }
  return 0;
}

static uint64_t memory_available_kb() {
  std::ifstream input("/proc/meminfo");
  std::string key;
  uint64_t value = 0;
  std::string unit;
  while (input >> key >> value >> unit) {
    if (key == "MemAvailable:") {
      return value;
    }
  }
  return 0;
}

static double max_temperature_c() {
  // Probe the 256 candidate zones once and remember the ones that exist.  This
  // runs inside the per-request timing window, where the unconditional probe
  // cost 256 open()/read() syscalls on every request.  Thermal zones do not
  // appear after boot, so a one-shot scan is sufficient; the server is
  // single-threaded (listen backlog 1), so the static needs no guard beyond the
  // C++11 thread-safe initialization it already gets.
  static const std::vector<std::string> zones = [] {
    std::vector<std::string> found;
    for (int i = 0; i < 256; ++i) {
      const std::string path =
          "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp";
      if (!read_text(path).empty()) {
        found.push_back(path);
      }
    }
    return found;
  }();
  double maximum = -1.0;
  for (const auto &path : zones) {
    const std::string value = read_text(path);
    if (value.empty()) {
      continue;
    }
    const double millidegrees = std::strtod(value.c_str(), nullptr);
    if (millidegrees > 0.0) {
      maximum = std::max(maximum, millidegrees / 1000.0);
    }
  }
  return maximum;
}

static int64_t optional_npu_value(const std::vector<std::string> &paths) {
  for (const auto &path : paths) {
    const std::string value = read_text(path);
    if (!value.empty()) {
      char *end = nullptr;
      const long long parsed = std::strtoll(value.c_str(), &end, 0);
      if (end != value.c_str()) {
        return parsed;
      }
    }
  }
  return -1;
}

struct TensorSpec {
  Qnn_Tensor_t tensor{};
  std::string name;
  std::vector<uint32_t> dimensions;
  std::vector<uint8_t> dynamic_dimensions;

  void copy_from(const Qnn_Tensor_t &source) {
    tensor = source;
    name = tensor_name(source) ? tensor_name(source) : "";
    const uint32_t rank = source.version == QNN_TENSOR_VERSION_2 ? source.v2.rank : source.v1.rank;
    const uint32_t *dims = source.version == QNN_TENSOR_VERSION_2 ? source.v2.dimensions
                                                                   : source.v1.dimensions;
    if (rank != 0) {
      dimensions.assign(dims, dims + rank);
    }
    if (source.version == QNN_TENSOR_VERSION_2 && source.v2.isDynamicDimensions) {
      dynamic_dimensions.assign(source.v2.isDynamicDimensions,
                                 source.v2.isDynamicDimensions + rank);
    }
    if (tensor.version == QNN_TENSOR_VERSION_2) {
      tensor.v2.name = name.c_str();
      tensor.v2.dimensions = dimensions.empty() ? nullptr : dimensions.data();
      tensor.v2.isDynamicDimensions = dynamic_dimensions.empty() ? nullptr
                                                                  : dynamic_dimensions.data();
    } else {
      tensor.v1.name = name.c_str();
      tensor.v1.dimensions = dimensions.empty() ? nullptr : dimensions.data();
    }
  }
};

struct GraphTiming {
  uint64_t load_us = 0;
  uint64_t load_count = 0;
  uint64_t cache_hits = 0;
  uint64_t tensor_prepare_us = 0;
  uint64_t output_allocate_us = 0;
  uint64_t wall_us = 0;
  uint64_t htp_accel_us = 0;
  uint64_t htp_rpc_us = 0;
  uint64_t htp_cycles = 0;
  struct ProfileEvent {
    uint32_t type = 0;
    uint32_t unit = 0;
    uint64_t value = 0;
    uint32_t depth = 0;
    int32_t parent = -1;
    std::string identifier;
  };
  std::vector<ProfileEvent> profile_events;
};

struct QnnRuntime {
  void *backend_library = nullptr;
  void *system_library = nullptr;
  const QNN_INTERFACE_VER_TYPE *qnn_provider = nullptr;
  const QNN_SYSTEM_INTERFACE_VER_TYPE *system_provider = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  bool supports_memhandle_tensors = false;
  bool supports_bind_mem_handles = false;
  QnnHtpDevice_PerfInfrastructure_t *htp_perf_infrastructure = nullptr;
  uint32_t htp_power_config_id = 0;

  ~QnnRuntime() {
    if (htp_perf_infrastructure && htp_power_config_id != 0 &&
        htp_perf_infrastructure->destroyPowerConfigId) {
      const auto status =
          htp_perf_infrastructure->destroyPowerConfigId(htp_power_config_id);
      if (status != QNN_SUCCESS) {
        std::fprintf(stderr, "warning: destroy HTP power config failed: 0x%x\n",
                     static_cast<unsigned>(status));
      }
    }
    if (backend && qnn_provider && qnn_provider->backendFree) {
      qnn_provider->backendFree(backend);
    }
    if (device && qnn_provider && qnn_provider->deviceFree) {
      qnn_provider->deviceFree(device);
    }
    if (backend_library) {
      dlclose(backend_library);
    }
    if (system_library) {
      dlclose(system_library);
    }
  }

  void initialize(const std::string &backend_path, const std::string &system_path) {
    backend_library = dlopen(backend_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!backend_library) {
      throw std::runtime_error("dlopen backend failed: " + std::string(dlerror()));
    }
    auto get_providers = reinterpret_cast<GetProvidersFn>(
        dlsym(backend_library, "QnnInterface_getProviders"));
    if (!get_providers) {
      throw std::runtime_error("QnnInterface_getProviders is missing");
    }
    const QnnInterface_t **providers = nullptr;
    uint32_t count = 0;
    check(get_providers(&providers, &count), "get QNN providers");
    if (!providers || count == 0) {
      throw std::runtime_error("QNN provider list is empty");
    }
    qnn_provider = &providers[0]->QNN_INTERFACE_VER_NAME;

    if (qnn_provider->propertyHasCapability) {
      const auto memhandle_status = qnn_provider->propertyHasCapability(
          QNN_PROPERTY_TENSOR_SUPPORT_MEMHANDLE_TYPE);
      const auto bind_status = qnn_provider->propertyHasCapability(
          QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_BIND_MEM_HANDLES);
      supports_memhandle_tensors = memhandle_status == QNN_PROPERTY_SUPPORTED;
      supports_bind_mem_handles = bind_status == QNN_PROPERTY_SUPPORTED;
      std::printf("QNN MEMHANDLE capability: tensors=0x%x bind_environment=0x%x\n",
                  static_cast<unsigned>(memhandle_status), static_cast<unsigned>(bind_status));
      std::fflush(stdout);
    } else {
      std::printf("QNN MEMHANDLE capability query is unavailable\n");
      std::fflush(stdout);
    }

    system_library = dlopen(system_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!system_library) {
      throw std::runtime_error("dlopen system failed: " + std::string(dlerror()));
    }
    auto get_system_providers = reinterpret_cast<GetSystemProvidersFn>(
        dlsym(system_library, "QnnSystemInterface_getProviders"));
    if (!get_system_providers) {
      throw std::runtime_error("QnnSystemInterface_getProviders is missing");
    }
    const QnnSystemInterface_t **system_providers = nullptr;
    uint32_t system_count = 0;
    check(get_system_providers(&system_providers, &system_count), "get QNN system providers");
    if (!system_providers || system_count == 0) {
      throw std::runtime_error("QNN system provider list is empty");
    }
    system_provider = &system_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    // Leave device selection to QNN, matching qnn-net-run on this board.
    check(qnn_provider->deviceCreate(nullptr, nullptr, &device), "deviceCreate");
    configure_htp_performance_from_environment();
    check(qnn_provider->backendCreate(nullptr, nullptr, &backend), "backendCreate");
    register_op_packages_from_environment();
  }

  static void check(Qnn_ErrorHandle_t status, const std::string &operation) {
    if (status != QNN_SUCCESS) {
      std::ostringstream error;
      error << operation << " failed: 0x" << std::hex << static_cast<unsigned>(status);
      throw std::runtime_error(error.str());
    }
  }

 private:
  void configure_htp_performance_from_environment() {
    const char *mode = std::getenv("XR1_HTP_PERF_MODE");
    if (!mode || !*mode) {
      return;
    }
    if (std::string(mode) != "high_performance") {
      throw std::runtime_error(
          "unsupported XR1_HTP_PERF_MODE; expected high_performance");
    }
    if (!qnn_provider->deviceGetInfrastructure) {
      throw std::runtime_error("QNN HTP performance infrastructure is unavailable");
    }

    QnnDevice_Infrastructure_t device_infrastructure = nullptr;
    check(qnn_provider->deviceGetInfrastructure(&device_infrastructure),
          "deviceGetInfrastructure");
    if (!device_infrastructure) {
      throw std::runtime_error("QNN returned a null HTP performance infrastructure");
    }
    auto *htp_infrastructure =
        static_cast<QnnHtpDevice_Infrastructure_t *>(device_infrastructure);
    if (htp_infrastructure->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
        !htp_infrastructure->perfInfra.createPowerConfigId ||
        !htp_infrastructure->perfInfra.setPowerConfig ||
        !htp_infrastructure->perfInfra.destroyPowerConfigId) {
      throw std::runtime_error("QNN HTP performance infrastructure is incomplete");
    }
    htp_perf_infrastructure = &htp_infrastructure->perfInfra;

    check(htp_perf_infrastructure->createPowerConfigId(0, 0, &htp_power_config_id),
          "create HTP power config");

    QnnHtpPerfInfrastructure_PowerConfig_t power_config{};
    power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto &dcvs = power_config.dcvsV3Config;
    dcvs.contextId = htp_power_config_id;
    dcvs.setDcvsEnable = 1;
    dcvs.dcvsEnable = 0;
    dcvs.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.setSleepLatency = 1;
    dcvs.sleepLatency = 40;
    dcvs.setSleepDisable = 1;
    dcvs.sleepDisable = 1;
    dcvs.setBusParams = 1;
    dcvs.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.setCoreParams = 1;
    dcvs.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    const QnnHtpPerfInfrastructure_PowerConfig_t *configs[] = {&power_config, nullptr};
    check(htp_perf_infrastructure->setPowerConfig(htp_power_config_id, configs),
          "set HTP high-performance power config");
    std::printf("HTP performance vote enabled: high_performance, power_config=%u\n",
                htp_power_config_id);
    std::fflush(stdout);
  }

  void register_op_packages_from_environment() {
    const char *encoded_packages = std::getenv("XR1_OP_PACKAGES");
    if (!encoded_packages || !*encoded_packages) {
      return;
    }
    if (!qnn_provider->backendRegisterOpPackage) {
      throw std::runtime_error("QNN backend does not support op-package registration");
    }

    std::string packages(encoded_packages);
    size_t entry_begin = 0;
    while (entry_begin < packages.size()) {
      const size_t entry_end = packages.find(',', entry_begin);
      const std::string entry = packages.substr(
          entry_begin, entry_end == std::string::npos ? std::string::npos : entry_end - entry_begin);
      const size_t first_separator = entry.find(':');
      const size_t second_separator =
          first_separator == std::string::npos ? std::string::npos : entry.find(':', first_separator + 1);
      if (entry.empty() || first_separator == std::string::npos ||
          second_separator == std::string::npos || entry.find(':', second_separator + 1) != std::string::npos ||
          first_separator == 0 || second_separator == first_separator + 1 ||
          second_separator + 1 == entry.size()) {
        throw std::runtime_error(
            "invalid XR1_OP_PACKAGES entry; expected path:interface_provider:CPU-or-HTP");
      }

      const std::string package_path = entry.substr(0, first_separator);
      const std::string provider =
          entry.substr(first_separator + 1, second_separator - first_separator - 1);
      const std::string target = entry.substr(second_separator + 1);
      check(qnn_provider->backendRegisterOpPackage(backend, package_path.c_str(), provider.c_str(),
                                                   target.c_str()),
            "backendRegisterOpPackage " + package_path + " target=" + target);
      std::printf("registered QNN op package target=%s path=%s\n", target.c_str(),
                  package_path.c_str());
      std::fflush(stdout);

      if (entry_end == std::string::npos) {
        break;
      }
      entry_begin = entry_end + 1;
    }
  }
};

class GraphModel {
 public:
  GraphModel(QnnRuntime *runtime, std::string name, std::string path,
             std::vector<std::string> logical_inputs,
             std::vector<std::string> logical_outputs,
             bool precision_compensation, uint32_t profile_level,
             bool capture_profile_events)
      : runtime_(runtime), name_(std::move(name)), path_(std::move(path)),
        logical_inputs_(std::move(logical_inputs)),
        logical_outputs_(std::move(logical_outputs)),
        precision_compensation_(precision_compensation),
        profile_level_(profile_level), capture_profile_events_(capture_profile_events) {}

  ~GraphModel() {
    release_dma_execution_environment();
    for (const auto &item : dma_handles_) {
      if (item.second) {
        runtime_->qnn_provider->memDeRegister(&item.second, 1);
      }
    }
    if (profile_ && runtime_->qnn_provider->profileFree) {
      runtime_->qnn_provider->profileFree(profile_);
    }
    if (context_ && runtime_->qnn_provider->contextFree) {
      runtime_->qnn_provider->contextFree(context_, nullptr);
    }
    if (mapped_ && mapped_size_ != 0) {
      munmap(mapped_, mapped_size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  void load() {
    fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error("open context failed: " + path_);
    }
    struct stat file_stat {};
    if (fstat(fd_, &file_stat) != 0 || file_stat.st_size <= 0) {
      throw std::runtime_error("invalid context size: " + path_);
    }
    mapped_size_ = static_cast<size_t>(file_stat.st_size);
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      throw std::runtime_error("mmap context failed: " + path_);
    }

    QnnSystemContext_Handle_t system_context = nullptr;
    QnnRuntime::check(runtime_->system_provider->systemContextCreate(&system_context),
                      "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
    QnnRuntime::check(runtime_->system_provider->systemContextGetMetaData(
                          system_context, mapped_, mapped_size_, &binary_info),
                      "systemContextGetMetaData");
    QnnSystemContext_GraphInfo_t *graphs = nullptr;
    uint32_t graph_count = 0;
    if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      graph_count = binary_info->contextBinaryInfoV3.numGraphs;
      graphs = binary_info->contextBinaryInfoV3.graphs;
    } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
      graph_count = binary_info->contextBinaryInfoV2.numGraphs;
      graphs = binary_info->contextBinaryInfoV2.graphs;
    } else {
      graph_count = binary_info->contextBinaryInfoV1.numGraphs;
      graphs = binary_info->contextBinaryInfoV1.graphs;
    }
    if (!graphs || graph_count == 0) {
      runtime_->system_provider->systemContextFree(system_context);
      throw std::runtime_error("context has no graphs: " + path_);
    }
    const QnnSystemContext_GraphInfoV1_t *graph_info = nullptr;
    for (uint32_t i = 0; i < graph_count; ++i) {
      if (graphs[i].graphInfoV1.graphName &&
          std::string(graphs[i].graphInfoV1.graphName) == name_) {
        graph_info = &graphs[i].graphInfoV1;
        break;
      }
    }
    if (!graph_info) {
      // Each exported context currently has one graph. Keep the actual graph
      // name from metadata so retrieve works even when the compiler hash changes.
      graph_info = &graphs[0].graphInfoV1;
    }
    actual_graph_name_ = graph_info->graphName ? graph_info->graphName : "";
    for (uint32_t i = 0; i < graph_info->numGraphInputs; ++i) {
      TensorSpec spec;
      spec.copy_from(graph_info->graphInputs[i]);
      inputs_.push_back(std::move(spec));
    }
    for (uint32_t i = 0; i < graph_info->numGraphOutputs; ++i) {
      TensorSpec spec;
      spec.copy_from(graph_info->graphOutputs[i]);
      outputs_.push_back(std::move(spec));
    }
    std::printf("metadata %-24s inputs=%zu outputs=%zu\n", name_.c_str(), inputs_.size(),
                outputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      std::printf("  input[%zu]=%s\n", i, inputs_[i].name.c_str());
    }
    std::fflush(stdout);
    if (!logical_inputs_.empty()) {
      if (logical_inputs_.size() != inputs_.size()) {
        runtime_->system_provider->systemContextFree(system_context);
        throw std::runtime_error("logical input count mismatch: " + name_);
      }
      const auto has_logical_name = [this](const std::string &name) {
        return std::find(logical_inputs_.begin(), logical_inputs_.end(), name) !=
               logical_inputs_.end();
      };
      for (auto &input : inputs_) {
        // QNN preserves the ONNX producer suffix on hidden_states for the
        // non-first Qwen partitions. Keep metadata order and alias only that
        // producer-specific name to the chain's logical tensor name.
        if ((input.name.rfind("hidden_states_", 0) == 0 ||
             input.name.rfind("hidden_states.", 0) == 0) &&
            has_logical_name("hidden_states")) {
          input.name = "hidden_states";
        }
      }
      for (const auto &logical_name : logical_inputs_) {
        const bool present = std::any_of(
            inputs_.begin(), inputs_.end(), [&logical_name](const TensorSpec &input) {
              return input.name == logical_name;
            });
        if (!present) {
          runtime_->system_provider->systemContextFree(system_context);
          throw std::runtime_error("logical input missing from context: " + name_ +
                                   ": " + logical_name);
        }
      }
    }
    if (!logical_outputs_.empty()) {
      if (logical_outputs_.size() > outputs_.size()) {
        runtime_->system_provider->systemContextFree(system_context);
        throw std::runtime_error("logical output count mismatch: " + name_);
      }
      bool numbered_outputs = !outputs_.empty();
      for (const auto &output : outputs_) {
        numbered_outputs = numbered_outputs && output.name.rfind("output_", 0) == 0;
      }
      if (numbered_outputs) {
        // AI Hub may append scalar fusion-anchor outputs to quantized graphs.
        // Bind only output_0..N to the manifest's business outputs and retain
        // the remaining numbered tensors under their physical names.
        for (size_t i = 0; i < logical_outputs_.size(); ++i) {
          const std::string physical_name = "output_" + std::to_string(i);
          auto output = std::find_if(
              outputs_.begin(), outputs_.end(), [&physical_name](const TensorSpec &spec) {
                return spec.name == physical_name;
              });
          if (output == outputs_.end()) {
            runtime_->system_provider->systemContextFree(system_context);
            throw std::runtime_error("logical output missing from context: " + name_ +
                                     ": " + physical_name);
          }
          output->name = logical_outputs_[i];
        }
      } else {
        if (logical_outputs_.size() != outputs_.size()) {
          runtime_->system_provider->systemContextFree(system_context);
          throw std::runtime_error("unnumbered output count mismatch: " + name_);
        }
        for (size_t i = 0; i < outputs_.size(); ++i) {
          outputs_[i].name = logical_outputs_[i];
        }
      }
    }
    runtime_->system_provider->systemContextFree(system_context);

    QnnRuntime::check(runtime_->qnn_provider->contextCreateFromBinary(
                          runtime_->backend, runtime_->device, nullptr, mapped_, mapped_size_,
                          &context_, nullptr),
                      "contextCreateFromBinary " + path_);
    QnnRuntime::check(runtime_->qnn_provider->graphRetrieve(context_, actual_graph_name_.c_str(),
                                                            &graph_),
                      "graphRetrieve " + path_);
    if (runtime_->qnn_provider->profileCreate) {
      const auto status = runtime_->qnn_provider->profileCreate(
          runtime_->backend, profile_level_, &profile_);
      if (status != QNN_SUCCESS) {
        profile_ = nullptr;
      }
    }
    const char *continuous_profile_environment = std::getenv("XR1_PROFILE_CONTINUOUS");
    const bool use_continuous_profile =
        continuous_profile_environment && parse_decimal<uint32_t>(continuous_profile_environment, 0) != 0;
    if (profile_ && profile_level_ != QNN_PROFILE_LEVEL_BASIC && use_continuous_profile) {
      QnnGraph_Config_t profile_config = QNN_GRAPH_CONFIG_INIT;
      profile_config.option = QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE;
      profile_config.profileHandle = profile_;
      const QnnGraph_Config_t *configs[] = {&profile_config, nullptr};
      QnnRuntime::check(runtime_->qnn_provider->graphSetConfig(graph_, configs),
                        "graphSetConfig profile " + path_);
      continuous_profile_ = true;
    }
    if (precision_compensation_) {
      QnnHtpGraph_CustomConfig_t htp_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
      htp_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION;
      htp_config.precisionCompensation = true;
      QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
      graph_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_config.customConfig = &htp_config;
      const QnnGraph_Config_t *configs[] = {&graph_config, nullptr};
      QnnRuntime::check(runtime_->qnn_provider->graphSetConfig(graph_, configs),
                        "graphSetConfig precision compensation " + path_);
    }
    std::printf("loaded %-24s graph=%s inputs=%zu outputs=%zu size=%zu\n", name_.c_str(),
                actual_graph_name_.c_str(), inputs_.size(), outputs_.size(), mapped_size_);
    std::fflush(stdout);
  }

  void release_io_resources() {
    release_dma_execution_environment();
    for (const auto &item : dma_handles_) {
      QnnRuntime::check(runtime_->qnn_provider->memDeRegister(&item.second, 1),
                        "memDeRegister " + name_);
    }
    dma_handles_.clear();
    output_pool_.clear();
  }

  const std::string &name() const { return name_; }

  const std::vector<TensorSpec> &input_specs() const { return inputs_; }
  const std::vector<TensorSpec> &output_specs() const { return outputs_; }

  size_t output_bytes(const std::string &name) const {
    const auto found = std::find_if(outputs_.begin(), outputs_.end(),
                                    [&name](const TensorSpec &spec) {
                                      return spec.name == name;
                                    });
    if (found == outputs_.end()) {
      throw std::runtime_error(this->name_ + " missing output " + name);
    }
    return tensor_bytes(found->tensor);
  }

  size_t input_bytes(const std::string &name) const {
    const auto found = std::find_if(inputs_.begin(), inputs_.end(), [&name](const TensorSpec &spec) {
      return spec.name == name;
    });
    if (found == inputs_.end()) {
      throw std::runtime_error(this->name_ + " missing input " + name);
    }
    return tensor_bytes(found->tensor);
  }

  Qnn_MemHandle_t dma_handle_for(const ByteBuffer &buffer, const TensorSpec &spec) {
    if (!buffer.dma_allocation()) {
      return nullptr;
    }
    const DmaHandleKey key{buffer.dma_allocation().get(), buffer.dma_offset(), spec.name};
    const auto found = dma_handles_.find(key);
    if (found != dma_handles_.end()) {
      return found->second;
    }
    if (!runtime_->qnn_provider->memRegister) {
      throw std::runtime_error(name_ + " QNN backend does not expose memRegister");
    }
    QnnMemHtp_Descriptor_t htp_descriptor =
        {QNN_HTP_MEM_SHARED_BUFFER, buffer.dma_allocation()->size(), {0}};
    htp_descriptor.sharedBufferConfig.fd = buffer.dma_allocation()->fd();
    htp_descriptor.sharedBufferConfig.offset = buffer.dma_offset();
    Qnn_MemDescriptor_t descriptor = {
        {spec.tensor.version == QNN_TENSOR_VERSION_2 ? spec.tensor.v2.rank : spec.tensor.v1.rank,
         const_cast<uint32_t *>(spec.tensor.version == QNN_TENSOR_VERSION_2
                                    ? spec.tensor.v2.dimensions
                                    : spec.tensor.v1.dimensions),
         nullptr},
        tensor_dtype(spec.tensor),
        QNN_MEM_TYPE_CUSTOM,
        {{-1}}};
    descriptor.customInfo = &htp_descriptor;
    Qnn_MemHandle_t handle = nullptr;
    QnnRuntime::check(runtime_->qnn_provider->memRegister(context_, &descriptor, 1, &handle),
                      "memRegister " + name_ + " " + spec.name);
    dma_handles_.emplace(key, handle);
    return handle;
  }

  std::map<std::string, ByteBuffer> execute(const std::map<std::string, ByteBuffer> &provided,
                                             GraphTiming *timing,
                                             const std::map<std::string, ByteBuffer> *bound_outputs =
                                                 nullptr) {
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    std::vector<ByteBuffer> output_data(outputs_.size());
    inputs.reserve(inputs_.size());
    outputs.reserve(outputs_.size());
    const uint64_t input_prepare_started = now_us();
    for (const auto &spec : inputs_) {
      auto found = provided.find(spec.name);
      if (found == provided.end()) {
        throw std::runtime_error(name_ + " missing input " + spec.name);
      }
      if (found->second.size() != tensor_bytes(spec.tensor)) {
        throw std::runtime_error(name_ + " input size mismatch " + spec.name);
      }
      Qnn_Tensor_t tensor = spec.tensor;
      if (found->second.dma_allocation()) {
        bind_tensor_memhandle(&tensor, QNN_TENSOR_TYPE_APP_WRITE,
                              dma_handle_for(found->second, spec));
      } else {
        bind_tensor(&tensor, QNN_TENSOR_TYPE_APP_WRITE,
                    const_cast<uint8_t *>(found->second.data()), found->second.size());
      }
      inputs.push_back(tensor);
    }
    timing->tensor_prepare_us += now_us() - input_prepare_started;
    const uint64_t output_allocate_started = now_us();
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const size_t output_size = tensor_bytes(outputs_[i].tensor);
      if (bound_outputs) {
        const auto found = bound_outputs->find(outputs_[i].name);
        if (found != bound_outputs->end()) {
          if (found->second.size() != output_size) {
            throw std::runtime_error(name_ + " bound output size mismatch " + outputs_[i].name);
          }
          output_data[i] = found->second;
        }
      }
      if (output_data[i].empty()) {
        auto &pooled = output_pool_[outputs_[i].name];
        if (pooled.size() != output_size) {
          pooled = ByteBuffer(output_size);
        }
        output_data[i] = pooled;
      }
      Qnn_Tensor_t tensor = outputs_[i].tensor;
      if (output_data[i].dma_allocation()) {
        bind_tensor_memhandle(&tensor, QNN_TENSOR_TYPE_APP_READ,
                              dma_handle_for(output_data[i], outputs_[i]));
      } else {
        bind_tensor(&tensor, QNN_TENSOR_TYPE_APP_READ, output_data[i].data(), output_data[i].size());
      }
      outputs.push_back(tensor);
    }
    timing->output_allocate_us += now_us() - output_allocate_started;
    const bool has_dma_input = std::any_of(inputs.begin(), inputs.end(), [](const Qnn_Tensor_t &tensor) {
      return tensor.version == QNN_TENSOR_VERSION_2
                 ? tensor.v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE
                 : tensor.v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE;
    });
    const bool has_dma_output = std::any_of(outputs.begin(), outputs.end(), [](const Qnn_Tensor_t &tensor) {
      return tensor.version == QNN_TENSOR_VERSION_2
                 ? tensor.v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE
                 : tensor.v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE;
    });
    const bool all_dma = !inputs.empty() && !outputs.empty() &&
                         std::all_of(inputs.begin(), inputs.end(), [](const Qnn_Tensor_t &tensor) {
                           return tensor.version == QNN_TENSOR_VERSION_2
                                      ? tensor.v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE
                                      : tensor.v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE;
                         }) &&
                         std::all_of(outputs.begin(), outputs.end(), [](const Qnn_Tensor_t &tensor) {
                           return tensor.version == QNN_TENSOR_VERSION_2
                                      ? tensor.v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE
                                      : tensor.v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE;
                         });
    if ((has_dma_input || has_dma_output) && !all_dma) {
      throw std::runtime_error(name_ + " rejects partial MEMHANDLE I/O binding");
    }
    if (all_dma && runtime_->supports_bind_mem_handles) {
      prepare_dma_execution_environment(inputs, outputs);
    }
    const uint64_t started = now_us();
    QnnRuntime::check(runtime_->qnn_provider->graphExecute(
                          graph_, inputs.data(), static_cast<uint32_t>(inputs.size()), outputs.data(),
                          static_cast<uint32_t>(outputs.size()),
                          continuous_profile_ ? nullptr : profile_, nullptr),
                      "graphExecute " + name_);
    timing->wall_us += now_us() - started;
    read_profile(timing);
    std::map<std::string, ByteBuffer> result;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      result.emplace(outputs_[i].name, std::move(output_data[i]));
    }
    return result;
  }

 private:
  struct DmaHandleKey {
    const DmaAllocation *allocation = nullptr;
    size_t offset = 0;
    std::string tensor_name;

    bool operator<(const DmaHandleKey &other) const {
      return std::tie(allocation, offset, tensor_name) <
             std::tie(other.allocation, other.offset, other.tensor_name);
    }
  };

  static Qnn_MemHandle_t tensor_mem_handle(const Qnn_Tensor_t &tensor) {
    return tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.memHandle : tensor.v1.memHandle;
  }

  static bool same_dma_binding(const std::vector<Qnn_Tensor_t> &left,
                               const std::vector<Qnn_Tensor_t> &right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (size_t i = 0; i < left.size(); ++i) {
      if (tensor_mem_handle(left[i]) != tensor_mem_handle(right[i])) {
        return false;
      }
    }
    return true;
  }

  void prepare_dma_execution_environment(const std::vector<Qnn_Tensor_t> &inputs,
                                         const std::vector<Qnn_Tensor_t> &outputs) {
    if (!runtime_->supports_bind_mem_handles ||
        !runtime_->qnn_provider->graphPrepareExecutionEnvironment) {
      throw std::runtime_error(name_ + " backend does not support full MEMHANDLE binding");
    }
    if (dma_execution_environment_prepared_) {
      if (!same_dma_binding(inputs, dma_environment_inputs_) ||
          !same_dma_binding(outputs, dma_environment_outputs_)) {
        throw std::runtime_error(name_ + " MEMHANDLE layout changed after environment preparation");
      }
      return;
    }
    dma_environment_inputs_ = inputs;
    dma_environment_outputs_ = outputs;
    dma_execution_environment_ = {};
    dma_execution_environment_.option = QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_BIND_MEM_HANDLES;
    dma_execution_environment_.tensorSet = QNN_TENSOR_SET_INIT;
    dma_execution_environment_.tensorSet.v1.numInputs =
        static_cast<uint32_t>(dma_environment_inputs_.size());
    dma_execution_environment_.tensorSet.v1.inputs = dma_environment_inputs_.data();
    dma_execution_environment_.tensorSet.v1.numOutputs =
        static_cast<uint32_t>(dma_environment_outputs_.size());
    dma_execution_environment_.tensorSet.v1.outputs = dma_environment_outputs_.data();
    QnnGraph_ExecuteEnvironment_t *environments[] = {&dma_execution_environment_};
    QnnRuntime::check(runtime_->qnn_provider->graphPrepareExecutionEnvironment(
                          graph_, environments, 1),
                      "prepareExecutionEnvironment " + name_);
    dma_execution_environment_prepared_ = true;
    std::printf("prepared full MEMHANDLE environment for %s (%zu inputs, %zu outputs)\\n",
                name_.c_str(), dma_environment_inputs_.size(), dma_environment_outputs_.size());
    std::fflush(stdout);
  }

  void release_dma_execution_environment() {
    if (!dma_execution_environment_prepared_) {
      return;
    }
    if (runtime_ && runtime_->qnn_provider &&
        runtime_->qnn_provider->graphReleaseExecutionEnvironment) {
      const QnnGraph_ExecuteEnvironment_t *environments[] = {&dma_execution_environment_};
      runtime_->qnn_provider->graphReleaseExecutionEnvironment(graph_, environments, 1);
    }
    dma_execution_environment_prepared_ = false;
  }

  void read_profile(GraphTiming *timing) {
    if (!profile_ || !runtime_->qnn_provider->profileGetEvents ||
        !runtime_->qnn_provider->profileGetEventData) {
      return;
    }
    const QnnProfile_EventId_t *event_ids = nullptr;
    uint32_t event_count = 0;
    if (runtime_->qnn_provider->profileGetEvents(profile_, &event_ids, &event_count) != QNN_SUCCESS) {
      return;
    }
    for (uint32_t i = 0; i < event_count; ++i) {
      collect_profile_event(event_ids[i], 0, -1, timing);
    }
  }

  void collect_profile_event(QnnProfile_EventId_t event_id, uint32_t depth, int32_t parent,
                             GraphTiming *timing) {
    QnnProfile_EventData_t event_data{};
    if (runtime_->qnn_provider->profileGetEventData(event_id, &event_data) != QNN_SUCCESS) {
      return;
    }
    switch (event_data.type) {
      case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC:
        timing->htp_accel_us = event_data.value;
        break;
      case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC:
        timing->htp_rpc_us = event_data.value;
        break;
      case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE:
        timing->htp_cycles = event_data.value;
        break;
      default:
        break;
    }
    int32_t event_index = -1;
    if (capture_profile_events_) {
      GraphTiming::ProfileEvent event;
      event.type = event_data.type;
      event.unit = event_data.unit;
      event.value = event_data.value;
      event.depth = depth;
      event.parent = parent;
      event.identifier = event_data.identifier ? event_data.identifier : "";
      timing->profile_events.push_back(std::move(event));
      event_index = static_cast<int32_t>(timing->profile_events.size() - 1);
    }
    if (!runtime_->qnn_provider->profileGetSubEvents) {
      return;
    }
    const QnnProfile_EventId_t *sub_event_ids = nullptr;
    uint32_t sub_event_count = 0;
    if (runtime_->qnn_provider->profileGetSubEvents(event_id, &sub_event_ids,
                                                    &sub_event_count) != QNN_SUCCESS) {
      return;
    }
    for (uint32_t i = 0; i < sub_event_count; ++i) {
      collect_profile_event(sub_event_ids[i], depth + 1, event_index, timing);
    }
  }

  QnnRuntime *runtime_;
  std::string name_;
  std::string path_;
  std::string actual_graph_name_;
  std::vector<std::string> logical_outputs_;
  int fd_ = -1;
  void *mapped_ = nullptr;
  size_t mapped_size_ = 0;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  Qnn_ProfileHandle_t profile_ = nullptr;
  std::vector<std::string> logical_inputs_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::map<std::string, ByteBuffer> output_pool_;
  std::map<DmaHandleKey, Qnn_MemHandle_t> dma_handles_;
  bool dma_execution_environment_prepared_ = false;
  QnnGraph_ExecuteEnvironment_t dma_execution_environment_{};
  std::vector<Qnn_Tensor_t> dma_environment_inputs_;
  std::vector<Qnn_Tensor_t> dma_environment_outputs_;
  bool precision_compensation_ = false;
  uint32_t profile_level_ = QNN_PROFILE_LEVEL_BASIC;
  bool capture_profile_events_ = false;
  bool continuous_profile_ = false;
};

struct Request {
  std::map<std::string, ByteBuffer> arrays;
};

static Request parse_request(const std::shared_ptr<std::string> &payload) {
  Request request;
  if (!payload) {
    throw std::runtime_error("null request payload");
  }
  const auto *data = reinterpret_cast<const uint8_t *>(payload->data());
  size_t offset = 0;
  if (payload->size() < 4) {
    throw std::runtime_error("request is too short");
  }
  const uint32_t count = read_u32_be(data);
  offset += 4;
  if (count > 128) {
    throw std::runtime_error("too many request arrays");
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (offset + 2 > payload->size()) {
      throw std::runtime_error("truncated request name length");
    }
    const uint16_t name_size = static_cast<uint16_t>((data[offset] << 8) | data[offset + 1]);
    offset += 2;
    if (name_size == 0 || offset + name_size + 8 > payload->size()) {
      throw std::runtime_error("truncated request array header");
    }
    const std::string name(reinterpret_cast<const char *>(data + offset), name_size);
    offset += name_size;
    const uint64_t size = read_u64_be(data + offset);
    offset += 8;
    if (size > kMaxFrameBytes || offset + size > payload->size()) {
      throw std::runtime_error("invalid request array size");
    }
    ByteBuffer value = ByteBuffer::from_payload(payload, offset, static_cast<size_t>(size));
    offset += value.size();
    request.arrays[name] = std::move(value);
  }
  if (offset != payload->size()) {
    throw std::runtime_error("trailing request bytes");
  }
  return request;
}

static const ByteBuffer &request_array(const Request &request, const std::string &prefix,
                                       const std::string &name) {
  const auto found = request.arrays.find(prefix + name);
  if (found == request.arrays.end()) {
    throw std::runtime_error("missing request array " + prefix + name);
  }
  return found->second;
}

static std::map<std::string, ByteBuffer> prefixed_state(const Request &request,
                                                        const std::string &prefix) {
  std::map<std::string, ByteBuffer> result;
  for (const auto &item : request.arrays) {
    if (item.first.rfind(prefix, 0) == 0) {
      result.emplace(item.first.substr(prefix.size()), item.second);
    }
  }
  return result;
}

struct RequestMetrics {
  uint64_t request_us = 0;
  uint64_t parse_us = 0;
  uint64_t vision_route_us = 0;
  uint64_t qwen_route_us = 0;
  uint64_t dit_route_us = 0;
  uint64_t dit_euler_us = 0;
  uint64_t response_us = 0;
  uint64_t host_copy_bytes = 0;
  uint64_t cpu_ticks = 0;
  uint64_t system_busy_ticks = 0;
  uint64_t system_total_ticks = 0;
  uint64_t rss_before_kb = 0;
  uint64_t rss_peak_kb = 0;
  uint64_t rss_after_kb = 0;
  uint64_t memory_available_kb = 0;
  double temperature_c = -1.0;
  uint32_t prefix_length = 0;
  std::map<std::string, GraphTiming> graphs;
};

static std::string json_number(uint64_t value) { return std::to_string(value); }

static std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (const char character : value) {
    switch (character) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << character;
        break;
    }
  }
  return out.str();
}

static std::string metrics_json(uint64_t request_index, const RequestMetrics &metrics) {
  std::ostringstream out;
  const double cpu_tick_hz = static_cast<double>(sysconf(_SC_CLK_TCK));
  const double cpu_percent = metrics.request_us == 0 || cpu_tick_hz <= 0.0
                                 ? 0.0
                                 : static_cast<double>(metrics.cpu_ticks) * 100000000.0 /
                                       (static_cast<double>(metrics.request_us) * cpu_tick_hz);
  // Real system-wide utilization: busy jiffies over total jiffies for the same
  // interval.  Taking both numbers from /proc/stat removes the dependency on
  // _SC_CLK_TCK and _SC_NPROCESSORS_ONLN agreeing with what the kernel
  // actually accounted for.  The key name is unchanged so existing log parsers
  // keep working; only the semantics are fixed.
  const double system_cpu_percent =
      metrics.system_total_ticks == 0
          ? 0.0
          : static_cast<double>(metrics.system_busy_ticks) * 100.0 /
                static_cast<double>(metrics.system_total_ticks);
  const double system_cpu_idle_percent =
      metrics.system_total_ticks == 0 ? 0.0 : 100.0 - system_cpu_percent;
  uint64_t total_htp_us = 0;
  uint64_t total_htp_rpc_us = 0;
  for (const auto &item : metrics.graphs) {
    total_htp_us += item.second.htp_accel_us;
    total_htp_rpc_us += item.second.htp_rpc_us;
  }
  const double htp_busy_estimate = metrics.request_us == 0
                                       ? 0.0
                                       : static_cast<double>(total_htp_us) * 100.0 /
                                             static_cast<double>(metrics.request_us);
  out << "{\"request\":" << request_index << ",\"request_us\":" << metrics.request_us
      << ",\"parse_us\":" << metrics.parse_us
      << ",\"vision_route_us\":" << metrics.vision_route_us
      << ",\"qwen_route_us\":" << metrics.qwen_route_us
      << ",\"dit_route_us\":" << metrics.dit_route_us
      << ",\"dit_euler_us\":" << metrics.dit_euler_us
      << ",\"response_us\":" << metrics.response_us
      << ",\"host_copy_bytes\":" << metrics.host_copy_bytes
      << ",\"cpu_process_percent_one_core\":" << cpu_percent
      << ",\"cpu_system_percent\":" << system_cpu_percent
      << ",\"cpu_system_idle_percent\":" << system_cpu_idle_percent
      << ",\"rss_before_kb\":" << metrics.rss_before_kb
      << ",\"rss_peak_kb\":" << metrics.rss_peak_kb
      << ",\"rss_after_kb\":" << metrics.rss_after_kb
      << ",\"mem_available_kb\":" << metrics.memory_available_kb
      << ",\"temperature_c\":" << metrics.temperature_c
      << ",\"prefix_length\":" << metrics.prefix_length
      << ",\"npu_util_source\":\"QNN_HTP_PROFILE\""
      << ",\"npu_accel_us\":" << total_htp_us
      << ",\"npu_rpc_us\":" << total_htp_rpc_us
      << ",\"npu_busy_percent_estimate\":" << htp_busy_estimate << ",\"graphs\":{";
  bool first = true;
  for (const auto &item : metrics.graphs) {
    if (!first) out << ',';
    first = false;
    out << '"' << item.first << "\":{\"wall_us\":" << item.second.wall_us
        << ",\"load_us\":" << item.second.load_us
        << ",\"load_count\":" << item.second.load_count
        << ",\"cache_hits\":" << item.second.cache_hits
        << ",\"tensor_prepare_us\":" << item.second.tensor_prepare_us
        << ",\"output_allocate_us\":" << item.second.output_allocate_us
        << ",\"htp_accel_us\":" << item.second.htp_accel_us
        << ",\"htp_rpc_us\":" << item.second.htp_rpc_us
        << ",\"htp_cycles\":" << item.second.htp_cycles;
    if (!item.second.profile_events.empty()) {
      out << ",\"profile_events\":[";
      for (size_t index = 0; index < item.second.profile_events.size(); ++index) {
        if (index != 0) {
          out << ',';
        }
        const auto &event = item.second.profile_events[index];
        out << "{\"type\":" << event.type << ",\"unit\":" << event.unit
            << ",\"value\":" << event.value << ",\"depth\":" << event.depth
            << ",\"parent\":" << event.parent
            << ",\"identifier\":\"" << json_escape(event.identifier) << "\"}";
      }
      out << ']';
    }
    out << '}';
  }
  out << "}}";
  return out.str();
}

class NativeServer {
 public:
  using GraphConfig =
      std::tuple<std::string, std::string, std::vector<std::string>,
                 std::vector<std::string>>;

  NativeServer(std::string model_root, std::string runtime_root, uint16_t port,
               std::string log_path, uint32_t qwen_layers_per_part,
               bool precision_compensation, std::string dump_dir,
               bool preload_models, bool release_graphs_after_request,
               bool release_qwen_graphs_after_execute, bool shared_dma_cache,
               bool full_dma_dit, bool full_dma_qwen, bool dit_flow5,
               uint32_t profile_level, std::string profile_graph)
      : model_root_(std::move(model_root)), runtime_root_(std::move(runtime_root)),
        port_(port), log_path_(std::move(log_path)),
        qwen_layers_per_part_(qwen_layers_per_part),
        precision_compensation_(precision_compensation), dump_dir_(std::move(dump_dir)),
        preload_models_(preload_models),
        release_graphs_after_request_(release_graphs_after_request),
        release_qwen_graphs_after_execute_(release_qwen_graphs_after_execute),
        shared_dma_cache_(shared_dma_cache),
        full_dma_dit_(full_dma_dit), full_dma_qwen_(full_dma_qwen), dit_flow5_(dit_flow5),
        profile_level_(profile_level), profile_graph_(std::move(profile_graph)) {
    const char *partial_environment = std::getenv("XR1_INITIAL_PRELOAD_GRAPHS");
    if (partial_environment && *partial_environment) {
      partial_preload_ = true;
      std::string value(partial_environment);
      size_t start = 0;
      while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        if (end > start) initial_preload_graphs_.push_back(value.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      if (initial_preload_graphs_.empty()) {
        throw std::runtime_error("XR1_INITIAL_PRELOAD_GRAPHS is empty");
      }
      vla_full_mode_ = false;
    } else {
      vla_full_mode_ = true;
    }
  }

  void initialize() {
    if (qwen_layers_per_part_ == 0 || 36u % qwen_layers_per_part_ != 0) {
      throw std::runtime_error("qwen layers per part must be a positive divisor of 36");
    }
    if (partial_preload_ && (!preload_models_ || release_graphs_after_request_)) {
      throw std::runtime_error("partial preload requires PRELOAD_MODELS=1 and no full request release");
    }
    const char *state_path = std::getenv("XR1_LIFECYCLE_STATE_PATH");
    lifecycle_state_path_ = state_path ? state_path : "";
    const std::string lib_dir = runtime_root_ + "/lib/aarch64-oe-linux-gcc11.2";
    runtime_.initialize(lib_dir + "/libQnnHtp.so", lib_dir + "/libQnnSystem.so");
    std::vector<std::string> vision_outputs = {
        "image_features", "deepstack_visual_embeds_0", "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2"};
    // The serialized StableGELU context was generated with a rotated output
    // order. Keep production's default ordering and make the experiment opt-in.
    const char *vision_output_order = std::getenv("XR1_VISION_OUTPUT_ORDER");
    if (vision_output_order && std::string(vision_output_order) == "stable_gelu_rotated") {
      vision_outputs = {
          "deepstack_visual_embeds_0", "deepstack_visual_embeds_1", "deepstack_visual_embeds_2",
          "image_features"};
      std::printf("StableGELU Vision output reorder enabled\n");
    }
    std::vector<GraphConfig> models = {
        {"vision", "models/vision.bin", {"pixel_values"}, vision_outputs},
        {"dit_step", "models/dit_step.bin",
         {"noisy_action", "timestep", "action_mask", "state", "cos", "sin",
          "attention_mask", "k_cache", "v_cache"},
         {"velocity"}},
    };
    if (dit_flow5_) {
      models.emplace_back(
          "dit_flow5", "models/dit_flow5.bin",
          std::vector<std::string>{"noisy_action", "timestep", "action_mask", "update_mask",
                                   "state", "cos", "sin", "attention_mask", "k_cache", "v_cache"},
          std::vector<std::string>{"raw_action"});
    }
    for (uint32_t start = 0; start < 36; start += qwen_layers_per_part_) {
      const uint32_t end = start + qwen_layers_per_part_;
      std::ostringstream name;
      name << "qwen_" << std::setfill('0') << std::setw(2) << start << "_"
           << std::setfill('0') << std::setw(2) << end;
      const std::string graph_name = name.str();
      qwen_chain_.push_back(graph_name);
      if (start == 0) {
        std::vector<std::string> inputs = {
            "input_ids", "attention_mask", "position_ids", "vision_embeds_padded",
            "visual_pos_mask", "deepstack_visual_embeds_0_padded"};
        if (end > 1) {
          inputs.push_back("deepstack_visual_embeds_1_padded");
        }
        if (end > 2) {
          inputs.push_back("deepstack_visual_embeds_2_padded");
        }
        models.insert(
            models.end() - 1,
            {graph_name, "models/" + graph_name + ".bin",
             std::move(inputs),
             {"hidden_states", "k_cache", "v_cache"}});
      } else {
        std::vector<std::string> inputs = {"hidden_states", "attention_mask", "position_ids"};
        if (start < 3) {
          inputs.push_back("deepstack_visual_embeds_" + std::to_string(start) + "_padded");
        }
        models.insert(models.end() - 1,
                      {graph_name, "models/" + graph_name + ".bin",
                       std::move(inputs),
                       {"hidden_states", "k_cache", "v_cache"}});
      }
    }
    for (const auto &model : models) {
      model_specs_[std::get<0>(model)] = model;
      model_load_order_.push_back(std::get<0>(model));
    }
    for (const auto &name : initial_preload_graphs_) {
      if (model_specs_.count(name) == 0) throw std::runtime_error("unknown resident graph: " + name);
      if (std::count(initial_preload_graphs_.begin(), initial_preload_graphs_.end(), name) != 1)
        throw std::runtime_error("duplicate resident graph: " + name);
    }
    publish_lifecycle_state("starting", "startup");
    if (preload_models_) {
      for (const auto &model : models) {
        const std::string &name = std::get<0>(model);
        if (!partial_preload_ || is_initial_preload_graph(name)) {
          load_graph(name);
        }
      }
      if (partial_preload_) {
        std::printf("preloaded %zu QNN contexts (partial VLA mode; send SIGUSR1 to enter full mode)\n",
                    loaded_graphs_.size());
      } else {
        std::printf("preloaded %zu QNN contexts\n", loaded_graphs_.size());
      }
      std::fflush(stdout);
    }
    if (release_graphs_after_request_) {
      std::printf("context release after each request enabled\n");
    }
    if (release_qwen_graphs_after_execute_) {
      std::printf("Qwen context release after each graph enabled\n");
    }
    if (full_dma_dit_) {
      std::printf("DiT full MEMHANDLE experiment enabled\n");
    }
    if (full_dma_qwen_) {
      std::printf("Qwen full MEMHANDLE experiment enabled\n");
    }
    if (partial_preload_) {
      std::printf("partial resident graphs: %s\n", join_graph_names(initial_preload_graphs_).c_str());
    }
    if (dit_flow5_) {
      std::printf("fused five-step DiT graph enabled\n");
    }
    if (profile_level_ != QNN_PROFILE_LEVEL_BASIC) {
      std::printf("diagnostic profile enabled level=%u graph=%s\n", profile_level_,
                  profile_graph_.empty() ? "<all>" : profile_graph_.c_str());
    }
    const auto slash = log_path_.find_last_of('/');
    if (slash != std::string::npos) {
      std::string directory = log_path_.substr(0, slash);
      mkdir(directory.c_str(), 0755);
    }
    if (!dump_dir_.empty()) {
      mkdir(dump_dir_.c_str(), 0755);
      std::printf("stage output dump enabled: %s\n", dump_dir_.c_str());
    }
    log_.open(log_path_, std::ios::out | std::ios::app);
    if (!log_) {
      throw std::runtime_error("cannot open metrics log " + log_path_);
    }
    cpu_ticks_before_ = process_cpu_ticks();
    system_ticks_before_ = system_cpu_ticks();
    std::printf("native XR1 QNN service ready on 0.0.0.0:%u\n", port_);
    std::fflush(stdout);
  }

  void serve() {
    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
      throw std::runtime_error("socket failed");
    }
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);
    if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(server, 1) != 0) {
      close(server);
      throw std::runtime_error("bind/listen failed");
    }
    publish_lifecycle_state("ready", "startup");
    while (true) {
      apply_lifecycle_requests();
      pollfd event{server, POLLIN, 0};
      const int polled = poll(&event, 1, 100);
      if (polled < 0 && errno != EINTR) throw std::runtime_error("listener poll failed");
      if (polled <= 0 || !(event.revents & POLLIN)) continue;
      const int client = accept(server, nullptr, nullptr);
      if (client < 0) {
        apply_lifecycle_requests();
        continue;
      }
      serve_client(client);
      close(client);
    }
  }

 private:
  bool is_initial_preload_graph(const std::string &name) const {
    return std::find(initial_preload_graphs_.begin(), initial_preload_graphs_.end(), name) !=
           initial_preload_graphs_.end();
  }

  static std::string join_graph_names(const std::vector<std::string> &names) {
    std::ostringstream out;
    for (size_t i = 0; i < names.size(); ++i) {
      if (i != 0) out << ',';
      out << names[i];
    }
    return out.str();
  }

  void publish_lifecycle_state(const std::string &phase, const std::string &operation,
                               const std::string &error = "", uint64_t duration_us = 0) {
    if (!partial_preload_ || lifecycle_state_path_.empty()) return;
    std::ostringstream out;
    out << "{\"pid\":" << getpid() << ",\"port\":" << port_
        << ",\"mode\":\"" << (vla_full_mode_ ? "full" : "partial")
        << "\",\"phase\":\"" << phase << "\",\"operation\":\"" << operation
        << "\",\"transition_id\":" << lifecycle_transition_id_
        << ",\"duration_ms\":" << duration_us / 1000.0
        << ",\"rss_kb\":" << rss_kb() << ",\"mem_available_kb\":" << memory_available_kb()
        << ",\"error\":\"" << json_escape(error) << "\",\"loaded_graphs\":[";
    bool first = true;
    for (const auto &item : loaded_graphs_) {
      if (!first) out << ',';
      out << '\"' << json_escape(item.first) << '\"'; first = false;
    }
    out << "]}\n";
    const std::string temporary = lifecycle_state_path_ + ".tmp";
    std::ofstream file(temporary, std::ios::trunc);
    file << out.str(); file.close();
    if (!file || std::rename(temporary.c_str(), lifecycle_state_path_.c_str()) != 0)
      throw std::runtime_error("cannot publish lifecycle state: " + lifecycle_state_path_);
  }

  void release_task_resources() {
    // All request-local views have gone out of scope. Even retained contexts
    // registered views into the shared cache, so detach EVERY graph first.
    for (auto &item : loaded_graphs_) item.second->release_io_resources();
    for (auto it = loaded_graphs_.begin(); it != loaded_graphs_.end();) {
      if (is_initial_preload_graph(it->first)) ++it;
      else it = loaded_graphs_.erase(it);
    }
    full_dma_qwen_buffers_.clear();
    shared_dma_dit_allocation_.reset();
    shared_dma_cache_allocation_.reset();
    padded_vision_buffers_.clear();
    reusable_k_cache_ = ByteBuffer();
    reusable_v_cache_ = ByteBuffer();
  }

  void enter_full_vla_mode() {
    if (!partial_preload_) return;
    const uint64_t started = now_us();
    ++lifecycle_transition_id_;
    publish_lifecycle_state("loading", "enter");
    try {
      for (const auto &name : model_load_order_) load_graph(name);
      vla_full_mode_ = true;
      std::printf("VLA full mode ready; preloaded %zu QNN contexts\n", loaded_graphs_.size());
      publish_lifecycle_state("ready", "enter", "", now_us() - started);
    } catch (const std::exception &error) {
      release_task_resources();
      vla_full_mode_ = false;
      publish_lifecycle_state("ready", "enter", error.what(), now_us() - started);
      std::fprintf(stderr, "VLA enter failed, restored partial mode: %s\n", error.what());
    }
    std::fflush(stdout);
  }

  void exit_full_vla_mode() {
    if (!partial_preload_) return;
    const uint64_t started = now_us();
    ++lifecycle_transition_id_;
    publish_lifecycle_state("unloading", "exit");
    release_task_resources();
    vla_full_mode_ = false;
    std::printf("VLA partial mode ready; retained %zu QNN contexts\n", loaded_graphs_.size());
    std::fflush(stdout);
    publish_lifecycle_state("ready", "exit", "", now_us() - started);
  }

  void apply_lifecycle_requests() {
    const std::sig_atomic_t request = g_lifecycle_request;
    if (request == 0) return;
    g_lifecycle_request = 0;
    if (request == 1) enter_full_vla_mode();
    if (request == 2) exit_full_vla_mode();
  }

  bool receive_with_lifecycle(int fd, void *buffer, size_t size) {
    auto *cursor = static_cast<uint8_t *>(buffer);
    while (size != 0) {
      // Finite poll also handles an idle persistent bridge connection. Signal
      // handlers only set a flag; all QNN calls stay on this service thread.
      apply_lifecycle_requests();
      pollfd event{fd, POLLIN, 0};
      const int polled = poll(&event, 1, 100);
      if (polled < 0 && errno == EINTR) continue;
      if (polled < 0) return false;
      if (polled == 0) continue;
      const ssize_t received = recv(fd, cursor, size, MSG_DONTWAIT);
      if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
      if (received <= 0) return false;
      cursor += received; size -= static_cast<size_t>(received);
    }
    return true;
  }

  void dump_outputs(uint64_t request_index, const std::string &graph,
                    const std::map<std::string, ByteBuffer> &outputs) {
    if (dump_dir_.empty()) {
      return;
    }
    for (const auto &item : outputs) {
      std::ostringstream path;
      path << dump_dir_ << "/request_" << std::setfill('0') << std::setw(4)
           << request_index << "_" << graph << "_" << item.first << ".f32";
      std::ofstream output(path.str(), std::ios::binary | std::ios::trunc);
      if (!output) {
        throw std::runtime_error("cannot write stage dump " + path.str());
      }
      output.write(reinterpret_cast<const char *>(item.second.data()),
                   static_cast<std::streamsize>(item.second.size()));
      if (!output) {
        throw std::runtime_error("short write for stage dump " + path.str());
      }
    }
  }

  std::map<std::string, ByteBuffer> execute_request(const Request &request,
                                                    RequestMetrics *metrics,
                                                    uint64_t request_index) {
    const char *allow_partial_environment = std::getenv("XR1_ALLOW_PARTIAL_INFERENCE");
    const bool allow_partial_inference =
        allow_partial_environment && parse_decimal<uint32_t>(allow_partial_environment, 0) != 0;
    if (partial_preload_ && !vla_full_mode_ && !allow_partial_inference) {
      throw std::runtime_error("VLA mode is partial; send SIGUSR1 before inference");
    }
    auto vision_outputs = prefixed_state(request, "vision_override__");
    if (vision_outputs.empty()) {
      auto vision_inputs = prefixed_state(request, "vision__");
      vision_outputs = execute_graph("vision", vision_inputs, metrics);
    }
    dump_outputs(request_index, "vision", vision_outputs);
    const uint64_t vision_route_started = now_us();
    auto state = prefixed_state(request, "qwen__");
    const ByteBuffer &mask = state.at("visual_pos_mask");
    const size_t mask_count = mask.size() / sizeof(float);
    const size_t hidden_size = 2560;
    for (const auto &mapping : std::vector<std::pair<std::string, std::string>>{
             {"image_features", "vision_embeds_padded"},
             {"deepstack_visual_embeds_0", "deepstack_visual_embeds_0_padded"},
             {"deepstack_visual_embeds_1", "deepstack_visual_embeds_1_padded"},
             {"deepstack_visual_embeds_2", "deepstack_visual_embeds_2_padded"}}) {
      const auto &source = vision_outputs.at(mapping.first);
      // These four tensors are all-zero except at visual-token positions, so a
      // client may omit them from the request frame -- 11,796,660 bytes at
      // seq 288, about 71% of it -- and let the server materialize them here.
      // Clients that still send them keep the original zero-copy path, which is
      // what makes it safe to deploy this server before updating any bridge.
      //
      // Zeroing is mandatory, not defensive: DeepStack layers 0-2 are injected
      // with an unmasked `hidden_states + visual` add (see
      // xr1_port/graph_models.py), so text-token rows must be exactly 0.  And
      // visual_pos_mask is derived from input_ids per request, so rows written
      // by the previous request would otherwise leak into this one.
      auto target_entry = state.find(mapping.second);
      if (target_entry == state.end()) {
        const size_t padded_bytes = mask_count * hidden_size * sizeof(float);
        ByteBuffer &scratch = padded_vision_buffers_[mapping.second];
        if (scratch.size() != padded_bytes) {
          scratch = ByteBuffer(padded_bytes);
        }
        std::memset(scratch.data(), 0, scratch.size());
        target_entry = state.emplace(mapping.second, scratch).first;
      }
      auto &target = target_entry->second;
      if (target.size() != mask_count * hidden_size * sizeof(float)) {
        throw std::runtime_error("unexpected size for qwen__" + mapping.second);
      }
      const float *mask_values = reinterpret_cast<const float *>(mask.data());
      const float *source_values = reinterpret_cast<const float *>(source.data());
      float *target_values = reinterpret_cast<float *>(target.data());
      size_t source_index = 0;
      for (size_t token = 0; token < mask_count; ++token) {
        if (mask_values[token] != 0.0f) {
          std::memcpy(target_values + token * hidden_size,
                      source_values + source_index * hidden_size, hidden_size * sizeof(float));
          ++source_index;
        }
      }
      if (source_index * hidden_size * sizeof(float) != source.size()) {
        throw std::runtime_error("vision token count mismatch");
      }
      metrics->host_copy_bytes += source.size();
    }
    metrics->vision_route_us += now_us() - vision_route_started;

    const uint64_t qwen_route_started = now_us();
    GraphModel &first_qwen = load_graph(qwen_chain_.front());
    const size_t k_cache_layer_bytes = first_qwen.output_bytes("k_cache");
    const size_t v_cache_layer_bytes = first_qwen.output_bytes("v_cache");
    const size_t k_cache_bytes = k_cache_layer_bytes * qwen_chain_.size();
    const size_t v_cache_bytes = v_cache_layer_bytes * qwen_chain_.size();
    ByteBuffer k_cache;
    ByteBuffer v_cache;
    if (shared_dma_cache_ || full_dma_qwen_) {
      const size_t total_cache_bytes = k_cache_bytes + v_cache_bytes;
      if (!shared_dma_cache_allocation_ ||
          shared_dma_cache_allocation_->size() != total_cache_bytes) {
        shared_dma_cache_allocation_ = DmaAllocation::allocate(total_cache_bytes);
        std::printf("allocated shared DMA K/V cache: %zu bytes fd=%d\n", total_cache_bytes,
                    shared_dma_cache_allocation_->fd());
        std::fflush(stdout);
      }
      k_cache = ByteBuffer::dma_view(shared_dma_cache_allocation_, 0, k_cache_bytes);
      v_cache = ByteBuffer::dma_view(shared_dma_cache_allocation_, k_cache_bytes, v_cache_bytes);
    } else {
      // Reuse the host staging buffers across requests.  ByteBuffer(size) goes
      // through std::vector<uint8_t>(size), which zero-initializes: allocating
      // fresh ones meant first-touching ~20,700 pages (84,934,656 bytes) inside
      // the Qwen window on every request.  Every layer slice is fully written
      // by Qwen before DiT reads it, so no clearing is required.
      if (reusable_k_cache_.size() != k_cache_bytes) {
        reusable_k_cache_ = ByteBuffer(k_cache_bytes);
      }
      if (reusable_v_cache_.size() != v_cache_bytes) {
        reusable_v_cache_ = ByteBuffer(v_cache_bytes);
      }
      k_cache = reusable_k_cache_;
      v_cache = reusable_v_cache_;
    }
    for (size_t index = 0; index < qwen_chain_.size(); ++index) {
      const auto &name = qwen_chain_[index];
      GraphModel &qwen_graph = load_graph(name);
      std::map<std::string, ByteBuffer> qwen_outputs = {
          {"k_cache", ByteBuffer::view(k_cache, index * k_cache_layer_bytes, k_cache_layer_bytes)},
          {"v_cache", ByteBuffer::view(v_cache, index * v_cache_layer_bytes, v_cache_layer_bytes)},
      };
      std::map<std::string, ByteBuffer> outputs;
      if (full_dma_qwen_) {
        auto &buffers = full_dma_qwen_buffers_for(qwen_graph);
        std::map<std::string, ByteBuffer> qwen_inputs;
        for (const auto &spec : qwen_graph.input_specs()) {
          const auto found = state.find(spec.name);
          if (found == state.end()) {
            throw std::runtime_error(name + " missing full-DMA input " + spec.name);
          }
          if (found->second.dma_allocation()) {
            qwen_inputs.emplace(spec.name, found->second);
          } else {
            auto target = buffers.inputs.at(spec.name);
            if (target.size() != found->second.size()) {
              throw std::runtime_error(name + " full-DMA input size mismatch " + spec.name);
            }
            std::memcpy(target.data(), found->second.data(), target.size());
            metrics->host_copy_bytes += target.size();
            qwen_inputs.emplace(spec.name, target);
            // Reuse common Qwen input buffers in following graph partitions.
            state[spec.name] = target;
          }
        }
        for (const auto &spec : qwen_graph.output_specs()) {
          if (spec.name != "k_cache" && spec.name != "v_cache") {
            qwen_outputs.emplace(spec.name, buffers.outputs.at(spec.name));
          }
        }
        outputs = execute_graph(name, qwen_inputs, metrics, &qwen_outputs);
      } else {
        outputs = execute_graph(name, state, metrics, &qwen_outputs);
      }
      dump_outputs(request_index, name, outputs);
      for (auto &item : outputs) {
        if (item.first != "k_cache" && item.first != "v_cache") {
          state[item.first] = std::move(item.second);
        }
      }
      if (release_qwen_graphs_after_execute_ && !is_initial_preload_graph(name)) {
        loaded_graphs_.erase(name);
      }
    }
    state["k_cache"] = std::move(k_cache);
    state["v_cache"] = std::move(v_cache);
    metrics->qwen_route_us += now_us() - qwen_route_started;

    const uint64_t dit_route_started = now_us();
    ByteBuffer action = request_array(request, "dit__", "noisy_action");
    uint32_t prefix_length = 0;
    const auto prefix = request.arrays.find("dit__prefix_length");
    if (prefix != request.arrays.end()) {
      if (prefix->second.size() != sizeof(uint32_t)) {
        throw std::runtime_error("dit__prefix_length must be one uint32");
      }
      std::memcpy(&prefix_length, prefix->second.data(), sizeof(prefix_length));
    }
    if (prefix_length > 30) {
      throw std::runtime_error("dit__prefix_length exceeds action horizon");
    }
    metrics->prefix_length = prefix_length;
    std::map<std::string, ByteBuffer> dit_inputs;
    dit_inputs["k_cache"] = state.at("k_cache");
    dit_inputs["v_cache"] = state.at("v_cache");
    for (const auto &key : {"action_mask", "state", "cos", "sin", "attention_mask"}) {
      dit_inputs[key] = request_array(request, "dit__", key);
    }
    ByteBuffer timestep_bytes(sizeof(float));
    std::map<std::string, ByteBuffer> dit_dma_inputs;
    std::map<std::string, ByteBuffer> dit_dma_outputs;
    if (full_dma_dit_) {
      GraphModel &dit_graph = load_graph("dit_step");
      const std::vector<std::string> dit_input_names = {
          "noisy_action", "timestep", "action_mask", "state", "cos", "sin",
          "attention_mask", "k_cache", "v_cache"};
      size_t total_bytes = 0;
      std::map<std::string, size_t> offsets;
      for (const auto &input_name : dit_input_names) {
        const auto source = dit_inputs.find(input_name);
        if (source != dit_inputs.end() && source->second.dma_allocation()) {
          continue;
        }
        total_bytes = align_up(total_bytes, kDmaAlignment);
        offsets[input_name] = total_bytes;
        total_bytes += dit_graph.input_bytes(input_name);
      }
      total_bytes = align_up(total_bytes, kDmaAlignment);
      offsets["velocity"] = total_bytes;
      total_bytes += dit_graph.output_bytes("velocity");
      total_bytes = align_up(total_bytes, kDmaAlignment);
      if (!shared_dma_dit_allocation_ || shared_dma_dit_allocation_->size() != total_bytes) {
        shared_dma_dit_allocation_ = DmaAllocation::allocate(total_bytes);
        std::printf("allocated full DiT DMA I/O: %zu bytes fd=%d\n", total_bytes,
                    shared_dma_dit_allocation_->fd());
        std::fflush(stdout);
      }
      for (const auto &input_name : dit_input_names) {
        const auto source = dit_inputs.find(input_name);
        if (source != dit_inputs.end() && source->second.dma_allocation()) {
          dit_dma_inputs.emplace(input_name, source->second);
        } else {
          dit_dma_inputs.emplace(input_name, ByteBuffer::dma_view(
              shared_dma_dit_allocation_, offsets.at(input_name), dit_graph.input_bytes(input_name)));
        }
      }
      dit_dma_outputs.emplace("velocity", ByteBuffer::dma_view(
          shared_dma_dit_allocation_, offsets.at("velocity"), dit_graph.output_bytes("velocity")));
      for (const auto &item : dit_inputs) {
        auto &destination = dit_dma_inputs.at(item.first);
        if (destination.data() != item.second.data()) {
          std::memcpy(destination.data(), item.second.data(), item.second.size());
          metrics->host_copy_bytes += item.second.size();
        }
      }
    }
    metrics->dit_route_us += now_us() - dit_route_started;
    if (dit_flow5_) {
      ByteBuffer update_mask(kActionBytes);
      float *mask_values = reinterpret_cast<float *>(update_mask.data());
      const size_t prefix_values = static_cast<size_t>(prefix_length) * 60u;
      for (size_t i = 0; i < kActionBytes / sizeof(float); ++i) {
        mask_values[i] = i < prefix_values ? 0.0f : 1.0f;
      }
      std::map<std::string, ByteBuffer> flow5_inputs = {
          {"noisy_action", action},
          {"timestep", timestep_bytes},
          {"action_mask", dit_inputs.at("action_mask")},
          {"update_mask", update_mask},
          {"state", dit_inputs.at("state")},
          {"cos", dit_inputs.at("cos")},
          {"sin", dit_inputs.at("sin")},
          {"attention_mask", dit_inputs.at("attention_mask")},
          {"k_cache", dit_inputs.at("k_cache")},
          {"v_cache", dit_inputs.at("v_cache")},
      };
      auto outputs = execute_graph("dit_flow5", flow5_inputs, metrics);
      return { {"raw_action", std::move(outputs.at("raw_action"))} };
    }
    for (uint32_t step = 0; step < kEulerSteps; ++step) {
      const uint64_t dit_step_started = now_us();
      dit_inputs["noisy_action"] = action;
      const float timestep = static_cast<float>(step) / static_cast<float>(kEulerSteps);
      std::memcpy(timestep_bytes.data(), &timestep, sizeof(timestep));
      dit_inputs["timestep"] = timestep_bytes;
      std::map<std::string, ByteBuffer> outputs;
      if (full_dma_dit_) {
        auto &dma_action = dit_dma_inputs.at("noisy_action");
        auto &dma_timestep = dit_dma_inputs.at("timestep");
        std::memcpy(dma_action.data(), action.data(), action.size());
        std::memcpy(dma_timestep.data(), timestep_bytes.data(), timestep_bytes.size());
        metrics->host_copy_bytes += action.size() + timestep_bytes.size();
        outputs = execute_graph("dit_step", dit_dma_inputs, metrics, &dit_dma_outputs);
      } else {
        outputs = execute_graph("dit_step", dit_inputs, metrics);
      }
      const auto &velocity = outputs.at("velocity");
      if (velocity.size() != action.size()) {
        throw std::runtime_error("DiT velocity/action size mismatch");
      }
      float *action_values = reinterpret_cast<float *>(action.data());
      const float *velocity_values = reinterpret_cast<const float *>(velocity.data());
      const size_t prefix_values = static_cast<size_t>(prefix_length) * 60u;
      for (size_t i = prefix_values; i < action.size() / sizeof(float); ++i) {
        action_values[i] += velocity_values[i] / static_cast<float>(kEulerSteps);
      }
      metrics->dit_euler_us += now_us() - dit_step_started;
    }
    if (action.size() != kActionBytes) {
      throw std::runtime_error("unexpected action size");
    }
    return {{"raw_action", std::move(action)}};
  }

  std::map<std::string, ByteBuffer> execute_graph(const std::string &name,
                                                  const std::map<std::string, ByteBuffer> &inputs,
                                                  RequestMetrics *metrics,
                                                  const std::map<std::string, ByteBuffer> *bound_outputs =
                                                      nullptr) {
    GraphTiming timing;
    const bool cache_hit = loaded_graphs_.find(name) != loaded_graphs_.end();
    const uint64_t load_started = now_us();
    GraphModel &graph = load_graph(name);
    timing.load_us = now_us() - load_started;
    timing.load_count = cache_hit ? 0 : 1;
    timing.cache_hits = cache_hit ? 1 : 0;
    metrics->rss_peak_kb = std::max(metrics->rss_peak_kb, rss_kb());
    auto outputs = graph.execute(inputs, &timing, bound_outputs);
    metrics->rss_peak_kb = std::max(metrics->rss_peak_kb, rss_kb());
    metrics->graphs[name].wall_us += timing.wall_us;
    metrics->graphs[name].load_us += timing.load_us;
    metrics->graphs[name].load_count += timing.load_count;
    metrics->graphs[name].cache_hits += timing.cache_hits;
    metrics->graphs[name].tensor_prepare_us += timing.tensor_prepare_us;
    metrics->graphs[name].output_allocate_us += timing.output_allocate_us;
    // These three were plain assignments while every neighbouring field
    // accumulates, so a graph executed N times per request (dit_step runs 5
    // Euler steps) only ever reported its last invocation.
    metrics->graphs[name].htp_accel_us += timing.htp_accel_us;
    metrics->graphs[name].htp_rpc_us += timing.htp_rpc_us;
    metrics->graphs[name].htp_cycles += timing.htp_cycles;
    metrics->graphs[name].profile_events = std::move(timing.profile_events);
    return outputs;
  }

  struct FullDmaGraphBuffers {
    std::shared_ptr<DmaAllocation> allocation;
    std::map<std::string, ByteBuffer> inputs;
    std::map<std::string, ByteBuffer> outputs;
  };

  FullDmaGraphBuffers &full_dma_qwen_buffers_for(GraphModel &graph) {
    auto &buffers = full_dma_qwen_buffers_[graph.name()];
    if (buffers.allocation) {
      return buffers;
    }
    size_t total_bytes = 0;
    std::map<std::string, size_t> input_offsets;
    std::map<std::string, size_t> output_offsets;
    for (const auto &spec : graph.input_specs()) {
      total_bytes = align_up(total_bytes, kDmaAlignment);
      input_offsets.emplace(spec.name, total_bytes);
      total_bytes += graph.input_bytes(spec.name);
    }
    for (const auto &spec : graph.output_specs()) {
      if (spec.name == "k_cache" || spec.name == "v_cache") {
        continue;
      }
      total_bytes = align_up(total_bytes, kDmaAlignment);
      output_offsets.emplace(spec.name, total_bytes);
      total_bytes += graph.output_bytes(spec.name);
    }
    total_bytes = align_up(total_bytes, kDmaAlignment);
    buffers.allocation = DmaAllocation::allocate(total_bytes);
    for (const auto &spec : graph.input_specs()) {
      buffers.inputs.emplace(spec.name, ByteBuffer::dma_view(
          buffers.allocation, input_offsets.at(spec.name), graph.input_bytes(spec.name)));
    }
    for (const auto &spec : graph.output_specs()) {
      if (spec.name != "k_cache" && spec.name != "v_cache") {
        buffers.outputs.emplace(spec.name, ByteBuffer::dma_view(
            buffers.allocation, output_offsets.at(spec.name), graph.output_bytes(spec.name)));
      }
    }
    std::printf("allocated full Qwen DMA I/O: %s %zu bytes fd=%d\n", graph.name().c_str(),
                total_bytes, buffers.allocation->fd());
    std::fflush(stdout);
    return buffers;
  }

  GraphModel &load_graph(const std::string &name) {
    const auto loaded = loaded_graphs_.find(name);
    if (loaded != loaded_graphs_.end()) {
      return *loaded->second;
    }
    const auto found = model_specs_.find(name);
    if (found == model_specs_.end()) {
      throw std::runtime_error("unknown graph " + name);
    }
    const auto &spec = found->second;
    auto graph = std::make_unique<GraphModel>(
        &runtime_, std::get<0>(spec), model_root_ + "/" + std::get<1>(spec),
        std::get<2>(spec), std::get<3>(spec), precision_compensation_,
        profile_level_ != QNN_PROFILE_LEVEL_BASIC &&
                (profile_graph_.empty() || profile_graph_ == name)
            ? profile_level_
            : QNN_PROFILE_LEVEL_BASIC,
        profile_level_ != QNN_PROFILE_LEVEL_BASIC &&
            (profile_graph_.empty() || profile_graph_ == name));
    graph->load();
    GraphModel *result = graph.get();
    loaded_graphs_.emplace(name, std::move(graph));
    return *result;
  }

  void release_loaded_graphs() {
    if (!loaded_graphs_.empty()) {
      loaded_graphs_.clear();
    }
  }

  std::string make_response(uint32_t status, const ByteBuffer &action,
                            const std::string &metrics) {
    std::string payload;
    append_u32_be(&payload, status);
    append_u32_be(&payload, static_cast<uint32_t>(action.size()));
    payload.append(reinterpret_cast<const char *>(action.data()), action.size());
    append_u32_be(&payload, static_cast<uint32_t>(metrics.size()));
    payload.append(metrics);
    std::string frame;
    append_u32_be(&frame, static_cast<uint32_t>(payload.size()));
    frame.append(payload);
    return frame;
  }

  void serve_client(int client) {
    while (true) {
      uint8_t size_bytes[4];
      if (!receive_with_lifecycle(client, size_bytes, sizeof(size_bytes))) {
        return;
      }
      const uint32_t frame_size = read_u32_be(size_bytes);
      if (frame_size == 0 || frame_size > kMaxFrameBytes) {
        return;
      }
      auto payload = std::make_shared<std::string>(frame_size, '\0');
      if (!receive_with_lifecycle(client, payload->data(), payload->size())) {
        return;
      }
      const uint64_t request_index = request_count_++;
      RequestMetrics metrics;
      const uint64_t started = now_us();
      metrics.rss_before_kb = rss_kb();
      metrics.rss_peak_kb = metrics.rss_before_kb;
      const uint64_t request_cpu_ticks_start = process_cpu_ticks();
      const SystemCpuTicks request_system_ticks_start = system_cpu_ticks();
      std::string response;
      ByteBuffer action;
      try {
        const uint64_t parse_started = now_us();
        const Request request = parse_request(payload);
        metrics.parse_us = now_us() - parse_started;
        auto result = execute_request(request, &metrics, request_index);
        action = std::move(result.at("raw_action"));
        metrics.cpu_ticks = process_cpu_ticks() - request_cpu_ticks_start;
        const SystemCpuTicks request_system_ticks_delta =
            system_cpu_ticks() - request_system_ticks_start;
        metrics.system_busy_ticks = request_system_ticks_delta.busy;
        metrics.system_total_ticks = request_system_ticks_delta.total;
        if (release_graphs_after_request_) {
          release_loaded_graphs();
        }
        metrics.rss_after_kb = rss_kb();
        metrics.memory_available_kb = memory_available_kb();
        metrics.temperature_c = max_temperature_c();
        metrics.request_us = now_us() - started;
        const uint64_t response_started = now_us();
        const std::string warmup_json = metrics_json(request_index, metrics);
        (void)make_response(0, action, warmup_json);
        metrics.response_us = now_us() - response_started;
        metrics.request_us = now_us() - started;
        const std::string json = metrics_json(request_index, metrics);
        log_ << json << '\n';
        log_.flush();
        response = make_response(0, action, json);
        std::printf("request %llu: %.3fs, HTP %.3fms, RSS %llukB, CPU %.1f%%\n",
                    static_cast<unsigned long long>(request_index), metrics.request_us / 1e6,
                    std::accumulate(metrics.graphs.begin(), metrics.graphs.end(), 0.0,
                                    [](double value, const auto &item) {
                                      return value + item.second.htp_accel_us / 1000.0;
                                    }),
                    static_cast<unsigned long long>(metrics.rss_after_kb),
                    metrics.request_us == 0
                        ? 0.0
                        : metrics.cpu_ticks * 100000000.0 /
                              (metrics.request_us * static_cast<double>(sysconf(_SC_CLK_TCK))));
        std::fflush(stdout);
      } catch (const std::exception &error) {
        metrics.request_us = now_us() - started;
        if (release_graphs_after_request_ || release_qwen_graphs_after_execute_) {
          release_loaded_graphs();
        }
        const std::string message = std::string("{\"error\":\"") + error.what() + "\"}";
        log_ << message << '\n';
        log_.flush();
        response = make_response(1, {}, message);
        std::fprintf(stderr, "request %llu failed: %s\n",
                     static_cast<unsigned long long>(request_index), error.what());
      }
      if (!send_all(client, response.data(), response.size())) {
        return;
      }
      apply_lifecycle_requests();
      cpu_ticks_before_ = process_cpu_ticks();
      system_ticks_before_ = system_cpu_ticks();
    }
  }

  std::string model_root_;
  std::string runtime_root_;
  uint16_t port_;
  std::string log_path_;
  uint32_t qwen_layers_per_part_;
  bool precision_compensation_;
  std::string dump_dir_;
  bool preload_models_;
  bool release_graphs_after_request_;
  bool release_qwen_graphs_after_execute_;
  bool shared_dma_cache_;
  bool full_dma_dit_;
  bool full_dma_qwen_;
  bool dit_flow5_;
  bool partial_preload_ = false;
  bool vla_full_mode_ = true;
  std::vector<std::string> initial_preload_graphs_;
  std::vector<std::string> model_load_order_;
  std::string lifecycle_state_path_;
  uint64_t lifecycle_transition_id_ = 0;
  uint32_t profile_level_ = QNN_PROFILE_LEVEL_BASIC;
  std::string profile_graph_;
  std::shared_ptr<DmaAllocation> shared_dma_cache_allocation_;
  std::shared_ptr<DmaAllocation> shared_dma_dit_allocation_;
  std::map<std::string, FullDmaGraphBuffers> full_dma_qwen_buffers_;
  // Zero-filled stand-ins for qwen__*_padded when the client omits them, and
  // reused host staging for the non-DMA K/V path.  Persistent so the pages stay
  // faulted in across requests.
  std::map<std::string, ByteBuffer> padded_vision_buffers_;
  ByteBuffer reusable_k_cache_;
  ByteBuffer reusable_v_cache_;
  QnnRuntime runtime_;
  std::map<std::string, GraphConfig> model_specs_;
  std::vector<std::string> qwen_chain_;
  std::map<std::string, std::unique_ptr<GraphModel>> loaded_graphs_;
  std::ofstream log_;
  uint64_t request_count_ = 0;
  uint64_t cpu_ticks_before_ = process_cpu_ticks();
  SystemCpuTicks system_ticks_before_ = system_cpu_ticks();
};

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: qnn_native_server MODEL_ROOT QAIRT_ROOT [PORT] [METRICS] [QWEN_LAYERS_PER_PART]\n";
    return 2;
  }
  const std::string model_root = argv[1];
  const std::string runtime_root = argv[2];
  const uint16_t port = argc > 3 ? parse_decimal<uint16_t>(argv[3], 10087) : 10087;
  const std::string log_path = argc > 4 ? argv[4] : "native_metrics.jsonl";
  const uint32_t qwen_layers_per_part =
      argc > 5 ? parse_decimal<uint32_t>(argv[5], 3) : 3;
  const bool precision_compensation =
      argc > 6 ? parse_decimal<uint32_t>(argv[6], 0) != 0 : false;
  const std::string dump_dir = argc > 7 ? argv[7] : "";
  const char *preload_environment = std::getenv("XR1_PRELOAD_MODELS");
  const bool preload_models =
      preload_environment && parse_decimal<uint32_t>(preload_environment, 0) != 0;
  const char *release_environment = std::getenv("XR1_RELEASE_GRAPHS_AFTER_REQUEST");
  const bool release_graphs_after_request =
      release_environment && parse_decimal<uint32_t>(release_environment, 0) != 0;
  const char *release_qwen_environment = std::getenv("XR1_RELEASE_QWEN_GRAPHS_AFTER_EXECUTE");
  const bool release_qwen_graphs_after_execute =
      release_qwen_environment && parse_decimal<uint32_t>(release_qwen_environment, 0) != 0;
  const char *shared_dma_environment = std::getenv("XR1_SHARED_DMA_CACHE");
  const bool shared_dma_cache =
      shared_dma_environment && parse_decimal<uint32_t>(shared_dma_environment, 0) != 0;
  const char *full_dma_dit_environment = std::getenv("XR1_DIT_FULL_DMA");
  const bool full_dma_dit =
      full_dma_dit_environment && parse_decimal<uint32_t>(full_dma_dit_environment, 0) != 0;
  const char *full_dma_qwen_environment = std::getenv("XR1_QWEN_FULL_DMA");
  const bool full_dma_qwen =
      full_dma_qwen_environment && parse_decimal<uint32_t>(full_dma_qwen_environment, 0) != 0;
  const char *dit_flow5_environment = std::getenv("XR1_DIT_FLOW5");
  const bool dit_flow5 =
      dit_flow5_environment && parse_decimal<uint32_t>(dit_flow5_environment, 0) != 0;
  const uint32_t profile_level = parse_profile_level(std::getenv("XR1_PROFILE_LEVEL"));
  const char *profile_graph_environment = std::getenv("XR1_PROFILE_GRAPH");
  const std::string profile_graph = profile_graph_environment ? profile_graph_environment : "vision";
  std::signal(SIGUSR1, handle_enter_vla_signal);
  std::signal(SIGUSR2, handle_exit_vla_signal);
  try {
    NativeServer server(model_root, runtime_root, port, log_path, qwen_layers_per_part,
                        precision_compensation, dump_dir, preload_models,
                        release_graphs_after_request, release_qwen_graphs_after_execute,
                        shared_dma_cache, full_dma_dit, full_dma_qwen, dit_flow5,
                        profile_level, profile_graph);
    server.initialize();
    server.serve();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "fatal: %s\n", error.what());
    return 1;
  }
  return 0;
}

import torch  # noqa: F401
import torch_npu  # noqa: F401

from flag_gems.runtime.backend.backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="ascend", device_name="npu", device_query_cmd="ascend"
)


CUSTOMIZED_UNUSED_OPS = ()
__all__ = ["*"]

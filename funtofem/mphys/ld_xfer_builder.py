from mphys import Builder
from .ld_xfer_components import DispXferComponent, LoadXferComponent


class TransferSchemeBuilder(Builder):
    def __init__(self):
        self.xfer = None
        self.ndof_struct = 0
        self.nnodes_struct = 0
        self.nnodes_aero = 0
        self.check_partials = False

    def initialize(self, comm):
        raise NotImplementedError("TransferSchemeBuilder initialize called directly")

    def get_coupling_group_subsystem(self, scenario_name=None):
        disp_xfer = DispXferComponent(
            xfer=self.xfer,
            struct_ndof=self.ndof_struct,
            struct_nnodes=self.nnodes_struct,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.check_partials,
        )

        load_xfer = LoadXferComponent(
            xfer=self.xfer,
            struct_ndof=self.ndof_struct,
            struct_nnodes=self.nnodes_struct,
            aero_nnodes=self.nnodes_aero,
            check_partials=self.check_partials,
        )

        return disp_xfer, load_xfer

from mphys import Builder
from funtofem import TransferScheme

from .ld_xfer_builder import TransferSchemeBuilder
from .mode_xfer_component import ModeTransfer


class MeldBuilder(TransferSchemeBuilder):
    def __init__(
        self,
        aero_builder: Builder,
        struct_builder: Builder,
        isym=-1,
        n=200,
        beta=0.5,
        check_partials=False,
    ):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder
        self.isym = isym
        self.n = n
        self.beta = beta
        self.check_partials = check_partials

    def initialize(self, comm):
        self.nnodes_aero = self.aero_builder.get_number_of_nodes()
        self.nnodes_struct = self.struct_builder.get_number_of_nodes()
        self.ndof_struct = self.struct_builder.get_ndof()

        self.xfer = TransferScheme.pyMELD(
            comm, comm, 0, comm, 0, self.isym, self.n, self.beta
        )


class MeldLfdBuilder(MeldBuilder):
    def __init__(
        self,
        aero_builder,
        struct_builder,
        nmodes,
        isym=-1,
        n=200,
        beta=0.5,
        check_partials=False,
    ):
        self.nmodes = nmodes
        super().__init__(aero_builder, struct_builder, isym, n, beta, check_partials)

    def get_pre_coupling_subsystem(self, scenario_name: str = None):
        return ModeTransfer(
            nmodes=self.nmodes,
            nnodes_struct=self.nnodes_struct,
            ndof_struct=self.ndof_struct,
            nnodes_aero=self.nnodes_aero,
            xfer=self.xfer,
        )

from funtofem import TransferScheme
from mphys import Builder

from .thermal_xfer_components import TemperatureXferComponent, HeatRateXferComponent

""" builder and components to wrap meld thermal to transfert temperature and
heat transfer rate between the convective and conductive analysis."""


class MeldThermalBuilder(Builder):
    def __init__(
        self,
        options,
        conv_builder: Builder,
        cond_builder: Builder,
        check_partials=False,
    ):
        super().__init__(options)
        self.check_partials = check_partials
        # TODO we can move the conv and cond builder to init_xfer_object call so that user does not need to worry about this
        self.conv_builder = conv_builder
        self.cond_builder = cond_builder

    # api level method for all builders
    def initialize(self, comm):
        # create the transfer
        self.xfer_object = TransferScheme.pyMELDThermal(
            comm,
            comm,
            0,
            comm,
            0,
            self.options["isym"],
            self.options["n"],
            self.options["beta"],
        )

        # TODO also do the necessary calls to the cond and conv builders to fully initialize MELD
        # for now, just save the counts
        self.cond_ndof = self.cond_builder.get_ndof()
        tacs = self.cond_builder.get_solver()
        get_surface = self.cond_builder.options["get_surface"]

        surface_nodes, mapping = get_surface(tacs)
        # get mapping of flow edge
        self.mapping = mapping
        self.cond_nnodes = len(mapping)

        self.conv_nnodes = self.conv_builder.get_nnodes(groupName="allIsothermalWalls")

    def get_coupling_group_subsystem(self, scenario_name=None):
        temp_xfer = TemperatureXferComponent(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials,
        )

        heat_xfer_xfer = HeatRateXferComponent(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials,
        )
        return temp_xfer, heat_xfer_xfer

import numpy as np
import openmdao.api as om
from funtofem import TransferScheme
from mphys import Builder

""" Components to wrap meld thermal to transfer temperature and
heat transfer rate between the convective and conductive analysis."""


class TemperatureXferComponent(om.ExplicitComponent):
    """
    Component to perform temperature transfer using MELD
    """

    def initialize(self):
        self.options.declare("xfer_object")
        self.options.declare("cond_ndof")
        self.options.declare("cond_nnodes")

        self.options.declare("conv_nnodes")
        self.options.declare("check_partials")
        self.options.declare("mapping")

        self.meldThermal = None
        self.initialized_meld = False

        self.cond_ndof = None
        self.cond_nnodes = None
        self.conv_nnodes = None
        self.check_partials = False

    def setup(self):
        self.meldThermal = self.options["xfer_object"]

        self.cond_ndof = self.options["cond_ndof"]
        self.cond_nnodes = self.options["cond_nnodes"]
        self.conv_nnodes = self.options["conv_nnodes"]
        self.check_partials = self.options["check_partials"]
        conv_nnodes = self.conv_nnodes

        # inputs
        self.add_input(
            "x_struct0",
            distributed=True,
            shape_by_conn=True,
            desc="initial structural node coordinates",
        )
        self.add_input(
            "x_aero0",
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
        )
        self.add_input(
            "T_conduct",
            distributed=True,
            shape_by_conn=True,
            desc="conductive node displacements",
        )

        # outputs
        print("T_convect", conv_nnodes)

        self.add_output(
            "T_convect",
            shape=conv_nnodes,
            distributed=True,
            val=np.ones(conv_nnodes) * 301,
            desc="conv surface temperatures",
        )

    def compute(self, inputs, outputs):

        x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
        mapping = self.options["mapping"]

        # x_surface =  np.zeros((len(mapping), 3))

        # for i in range(len(mapping)):
        #     idx = mapping[i]*3
        #     x_surface[i] = x_s0[idx:idx+3]

        self.meldThermal.setStructNodes(x_s0)
        self.meldThermal.setAeroNodes(x_a0)

        # heat_xfer_cond0 = np.array(inputs['heat_xfer_cond0'],dtype=TransferScheme.dtype)
        # heat_xfer_conv0 = np.array(inputs['heat_xfer_conv0'],dtype=TransferScheme.dtype)
        temp_conv = np.array(outputs["T_convect"], dtype=TransferScheme.dtype)

        temp_cond = np.array(inputs["T_conduct"], dtype=TransferScheme.dtype)
        # for i in range(3):
        #     temp_cond[i::3] = inputs['T_conduct'][i::self.cond_ndof]

        if not self.initialized_meld:
            self.meldThermal.initialize()
            self.initialized_meld = True

        self.meldThermal.transferTemp(temp_cond, temp_conv)

        outputs["T_convect"] = temp_conv


class HeatRateXferComponent(om.ExplicitComponent):
    """
    Component to perform transfers of heat rate using MELD
    """

    def initialize(self):
        self.options.declare("xfer_object")
        self.options.declare("cond_ndof")
        self.options.declare("cond_nnodes")

        self.options.declare("conv_nnodes")
        self.options.declare("check_partials")
        self.options.declare("mapping")

        self.meldThermal = None
        self.initialized_meld = False

        self.cond_ndof = None
        self.cond_nnodes = None
        self.conv_nnodes = None
        self.check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.meldThermal = self.options["xfer_object"]

        self.cond_ndof = self.options["cond_ndof"]
        self.cond_nnodes = self.options["cond_nnodes"]
        self.conv_nnodes = self.options["conv_nnodes"]
        self.check_partials = self.options["check_partials"]

        # inputs
        self.add_input(
            "x_struct0",
            distributed=True,
            shape_by_conn=True,
            desc="initial structural node coordinates",
        )
        self.add_input(
            "x_aero0",
            distributed=True,
            shape_by_conn=True,
            desc="initial aerodynamic surface node coordinates",
        )
        self.add_input(
            "q_convect",
            distributed=True,
            shape_by_conn=True,
            desc="initial conv heat transfer rate",
        )

        print("q_conduct", self.cond_nnodes)

        # outputs
        self.add_output(
            "q_conduct",
            distributed=True,
            shape=self.cond_nnodes,
            desc="heat transfer rate on the conduction mesh at the interface",
        )

    def compute(self, inputs, outputs):

        heat_xfer_conv = np.array(inputs["q_convect"], dtype=TransferScheme.dtype)
        heat_xfer_cond = np.zeros(self.cond_nnodes, dtype=TransferScheme.dtype)

        # if self.check_partials:
        #     x_s0 = np.array(inputs['x_struct0'],dtype=TransferScheme.dtype)
        #     x_a0 = np.array(inputs['x_aero0'],dtype=TransferScheme.dtype)
        #     self.meldThermal.setStructNodes(x_s0)
        #     self.meldThermal.setAeroNodes(x_a0)

        #     #TODO meld needs a set state rather requiring transferDisps to update the internal state

        #     temp_conv = np.zeros(inputs['q_convect'].size,dtype=TransferScheme.dtype)
        #     temp_cond  = np.zeros(self.cond_surface_nnodes,dtype=TransferScheme.dtype)
        #     for i in range(3):
        #         temp_cond[i::3] = inputs['T_conduct'][i::self.cond_ndof]

        #     self.meldThermal.transferTemp(temp_cond,temp_conv)

        self.meldThermal.transferFlux(heat_xfer_conv, heat_xfer_cond)
        outputs["q_conduct"] = heat_xfer_cond

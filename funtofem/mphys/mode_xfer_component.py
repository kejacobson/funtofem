import numpy as np
import openmdao.api as om

from funtofem import TransferScheme


class ModeTransfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nmodes")
        self.options.declare("nnodes_struct")
        self.options.declare("ndof_struct")
        self.options.declare("nnodes_aero")
        self.options.declare("xfer")
        self.options.declare(
            "use_ref_coordinates",
            types=bool,
            desc="Use a separate x_aero_ref, x_struct_ref for transfer scheme initialization",
        )

        self.initialized_xfer = True

    def setup(self):
        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "mode_shapes_struct",
            shape_by_conn=True,
            distributed=True,
            tags="mphys_coupling",
        )
        self.add_input(
            "x_aero0", shape_by_conn=True, distributed=True, tags=["mphys_coordinates"]
        )

        if self.options["use_ref_coordinates"]:
            self.add_input(
                "x_struct_ref",
                shape_by_conn=True,
                distributed=True,
                desc="initial structural node coordinates",
                tags=["mphys_coordinates"],
            )
            self.add_input(
                "x_aero_ref",
                shape_by_conn=True,
                distributed=True,
                desc="initial aero surface node coordinates",
                tags=["mphys_coordinates"],
            )

        nmodes = self.options["nmodes"]
        self.nnodes_aero = self.options["nnodes_aero"]
        self.nnodes_struct = self.options["nnodes_struct"]
        self.ndof_struct = self.options["ndof_struct"]

        aero_mode_size = (self.nnodes_aero * 3, nmodes)
        self.add_output(
            "mode_shapes_aero",
            shape=aero_mode_size,
            distributed=True,
            tags=["mphys_coupling"],
        )

    def _initialize_xfer(self, inputs):
        if self.options["use_ref_coodinates"]:
            x_s0 = np.array(inputs["x_struct_ref"], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs["x_aero_ref"], dtype=TransferScheme.dtype)
        else:
            x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)

        self.xfer.setStructNodes(x_s0)
        self.xfer.setAeroNodes(x_a0)
        self.xfer.initialize()
        self.initialized_xfer = True

    def compute(self, inputs, outputs):
        xfer = self.options["xfer"]
        nmodes = self.options["nmodes"]
        aero_X = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
        struct_X = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)

        aero_modes = np.zeros((aero_X.size, nmodes), dtype=TransferScheme.dtype)
        struct_modes = inputs["mode_shapes_struct"].reshape((-1, nmodes))

        if self.initialized_xfer:
            self._initialize_xfer(inputs)

        xfer.setAeroNodes(aero_X)
        xfer.setStructNodes(struct_X)

        struct_mode = np.zeros(self.nnodes_struct * 3, dtype=TransferScheme.dtype)
        for mode in range(nmodes):
            struct_mode[0::3] = struct_modes[0 :: self.ndof_struct, mode]
            struct_mode[1::3] = struct_modes[1 :: self.ndof_struct, mode]
            struct_mode[2::3] = struct_modes[2 :: self.ndof_struct, mode]
            aero_mode = np.zeros_like(aero_X)
            xfer.transferDisps(struct_mode, aero_mode)
            aero_modes[:, mode] = aero_mode

        outputs["mode_shapes_aero"] = aero_modes.copy()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        xfer = self.options["xfer"]
        nmodes = self.options["nmodes"]
        x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)

        xfer.setStructNodes(x_s0)
        xfer.setAeroNodes(x_a0)

        for imode in range(nmodes):
            u_s = np.zeros(self.nnodes_struct * 3, dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs["mode_shapes_struct"][i :: self.ndof_struct, imode]
            u_a = np.zeros(self.nnodes_aero * 3, dtype=TransferScheme.dtype)
            xfer.transferDisps(u_s, u_a)
            if mode == "fwd":
                if "mode_shapes_aero" in d_outputs:
                    if "mode_shapes_struct" in d_inputs:
                        d_in = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        for i in range(3):
                            d_in[i::3] = d_inputs["mode_shapes_struct"][
                                i :: self.ndof_struct, imode
                            ]
                        prod = np.zeros(
                            self.nnodes_aero * 3, dtype=TransferScheme.dtype
                        )
                        xfer.applydDduS(d_in, prod)
                        d_outputs["mode_shapes_aero"][:, imode] -= np.array(
                            prod, dtype=float
                        )
            if mode == "rev":
                if "mode_shapes_aero" in d_outputs:
                    du_a = np.array(
                        d_outputs["mode_shapes_aero"][:, imode],
                        dtype=TransferScheme.dtype,
                    )
                    if "mode_shapes_struct" in d_inputs:
                        # du_a/du_s^T * psi = - dD/du_s^T psi
                        prod = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        xfer.applydDduSTrans(du_a, prod)
                        for i in range(3):
                            d_inputs["mode_shapes_struct"][
                                i :: self.ndof_struct, imode
                            ] -= np.array(prod[i::3], dtype=np.float64)

                    # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                    if "x_aero0" in d_inputs:
                        prod = np.zeros(
                            d_inputs["x_aero0"].size, dtype=TransferScheme.dtype
                        )
                        xfer.applydDdxA0(du_a, prod)
                        d_inputs["x_aero0"] -= np.array(prod, dtype=float)

                    if "x_struct0" in d_inputs:
                        prod = np.zeros(
                            self.nnodes_struct * 3, dtype=TransferScheme.dtype
                        )
                        xfer.applydDdxS0(du_a, prod)
                        d_inputs["x_struct0"] -= np.array(prod, dtype=float)

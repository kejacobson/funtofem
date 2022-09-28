import numpy as np
import openmdao.api as om

from funtofem import TransferScheme


class DispXferComponent(om.ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """

    def initialize(self):
        self.options.declare("xfer", recordable=False)
        self.options.declare("struct_ndof")
        self.options.declare("struct_nnodes")
        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")
        self.options.declare(
            "use_ref_coordinates",
            types=bool,
            desc="Use a separate x_aero_ref, x_struct_ref for transfer scheme initialization",
        )

        self.xfer = None
        self.initialized_xfer = False

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.check_partials = False

    def setup(self):
        self.xfer = self.options["xfer"]

        self.struct_ndof = self.options["struct_ndof"]
        self.struct_nnodes = self.options["struct_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.check_partials = self.options["check_partials"]

        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        # inputs
        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            desc="initial structural node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "x_aero0",
            shape_by_conn=True,
            distributed=True,
            desc="initial aero surface node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "u_struct",
            shape_by_conn=True,
            distributed=True,
            desc="structural node displacements",
            tags=["mphys_coupling"],
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

        # outputs
        self.add_output(
            "u_aero",
            shape=self.aero_nnodes * 3,
            distributed=True,
            val=np.zeros(self.aero_nnodes * 3),
            desc="aerodynamic surface displacements",
            tags=["mphys_coupling"],
        )

        # partials
        # self.declare_partials('u_aero',['x_struct0','x_aero0','u_struct'])

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
        x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
        u_a = np.array(outputs["u_aero"], dtype=TransferScheme.dtype)

        u_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs["u_struct"][i :: self.struct_ndof]

        if not self.initialized_xfer:
            self._initialize_xfer(inputs)

        self.xfer.setStructNodes(x_s0)
        self.xfer.setAeroNodes(x_a0)

        self.xfer.transferDisps(u_s, u_a)

        outputs["u_aero"] = u_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """
        if self.check_partials:
            x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
            self.xfer.setStructNodes(x_s0)
            self.xfer.setAeroNodes(x_a0)
        u_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs["u_struct"][i :: self.struct_ndof]
        u_a = np.zeros(self.aero_nnodes * 3, dtype=TransferScheme.dtype)
        self.xfer.transferDisps(u_s, u_a)

        if mode == "fwd":
            if "u_aero" in d_outputs:
                if "u_struct" in d_inputs:
                    d_in = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    for i in range(3):
                        d_in[i::3] = d_inputs["u_struct"][i :: self.struct_ndof]
                    prod = np.zeros(self.aero_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydDduS(d_in, prod)
                    d_outputs["u_aero"] -= np.array(prod, dtype=float)

                if "x_aero0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError(
                            "XFER forward mode requested but not implemented"
                        )

                if "x_struct0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError(
                            "XFER forward mode requested but not implemented"
                        )

        if mode == "rev":
            if "u_aero" in d_outputs:
                du_a = np.array(d_outputs["u_aero"], dtype=TransferScheme.dtype)
                if "u_struct" in d_inputs:
                    # du_a/du_s^T * psi = - dD/du_s^T psi
                    prod = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydDduSTrans(du_a, prod)
                    for i in range(3):
                        d_inputs["u_struct"][i :: self.struct_ndof] -= np.array(
                            prod[i::3], dtype=np.float64
                        )

                # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                if "x_aero0" in d_inputs:
                    prod = np.zeros(
                        d_inputs["x_aero0"].size, dtype=TransferScheme.dtype
                    )
                    self.xfer.applydDdxA0(du_a, prod)
                    d_inputs["x_aero0"] -= np.array(prod, dtype=float)

                if "x_struct0" in d_inputs:
                    prod = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydDdxS0(du_a, prod)
                    d_inputs["x_struct0"] -= np.array(prod, dtype=float)


class LoadXferComponent(om.ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """

    def initialize(self):
        self.options.declare("xfer", recordable=False)
        self.options.declare("struct_ndof")
        self.options.declare("struct_nnodes")
        self.options.declare("aero_nnodes")
        self.options.declare("check_partials")

        self.xfer = None

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.xfer = self.options["xfer"]

        self.struct_ndof = self.options["struct_ndof"]
        self.struct_nnodes = self.options["struct_nnodes"]
        self.aero_nnodes = self.options["aero_nnodes"]
        self.check_partials = self.options["check_partials"]

        # self.set_check_partial_options(wrt='*',method='cs',directional=True)

        struct_ndof = self.struct_ndof
        struct_nnodes = self.struct_nnodes

        # inputs
        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            desc="initial structural node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "x_aero0",
            shape_by_conn=True,
            distributed=True,
            desc="initial aero surface node coordinates",
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "u_struct",
            shape_by_conn=True,
            distributed=True,
            desc="structural node displacements",
            tags=["mphys_coupling"],
        )
        self.add_input(
            "f_aero",
            shape_by_conn=True,
            distributed=True,
            desc="aerodynamic force vector",
            tags=["mphys_coupling"],
        )

        # outputs
        self.add_output(
            "f_struct",
            shape=struct_nnodes * struct_ndof,
            distributed=True,
            desc="structural force vector",
            tags=["mphys_coupling"],
        )

        # partials
        # self.declare_partials('f_struct',['x_struct0','x_aero0','u_struct','f_aero'])

    def compute(self, inputs, outputs):
        if self.check_partials:
            x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
            self.xfer.setStructNodes(x_s0)
            self.xfer.setAeroNodes(x_a0)
        f_a = np.array(inputs["f_aero"], dtype=TransferScheme.dtype)
        f_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)

        u_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs["u_struct"][i :: self.struct_ndof]
        u_a = np.zeros(inputs["f_aero"].size, dtype=TransferScheme.dtype)
        self.xfer.transferDisps(u_s, u_a)

        self.xfer.transferLoads(f_a, f_s)

        outputs["f_struct"][:] = 0.0
        for i in range(3):
            outputs["f_struct"][i :: self.struct_ndof] = f_s[i::3]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            f_s = g(f_a,u_s,x_a0,x_s0)
        The MELD internal residual is defined as:
            L = f_s - g(f_a,u_s,x_a0,x_s0)
        So explicit partials below for f_s are negative partials of L
        """
        if self.check_partials:
            x_s0 = np.array(inputs["x_struct0"], dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs["x_aero0"], dtype=TransferScheme.dtype)
            self.xfer.setStructNodes(x_s0)
            self.xfer.setAeroNodes(x_a0)
        f_a = np.array(inputs["f_aero"], dtype=TransferScheme.dtype)
        f_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)

        u_s = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs["u_struct"][i :: self.struct_ndof]
        u_a = np.zeros(inputs["f_aero"].size, dtype=TransferScheme.dtype)
        self.xfer.transferDisps(u_s, u_a)
        self.xfer.transferLoads(f_a, f_s)

        if mode == "fwd":
            if "f_struct" in d_outputs:
                if "u_struct" in d_inputs:
                    d_in = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    for i in range(3):
                        d_in[i::3] = d_inputs["u_struct"][i :: self.struct_ndof]
                    prod = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydLduS(d_in, prod)
                    for i in range(3):
                        d_outputs["f_struct"][i :: self.struct_ndof] -= np.array(
                            prod[i::3], dtype=float
                        )

                if "f_aero" in d_inputs:
                    # df_s/df_a psi = - dL/df_a * psi = -dD/du_s^T * psi
                    prod = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    df_a = np.array(d_inputs["f_aero"], dtype=TransferScheme.dtype)
                    self.xfer.applydDduSTrans(df_a, prod)
                    for i in range(3):
                        d_outputs["f_struct"][i :: self.struct_ndof] -= np.array(
                            prod[i::3], dtype=float
                        )

                if "x_aero0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

                if "x_struct0" in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError("forward mode requested but not implemented")

        if mode == "rev":
            if "f_struct" in d_outputs:
                d_out = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                for i in range(3):
                    d_out[i::3] = d_outputs["f_struct"][i :: self.struct_ndof]

                if "u_struct" in d_inputs:
                    d_in = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    # df_s/du_s^T * psi = - dL/du_s^T * psi
                    self.xfer.applydLduSTrans(d_out, d_in)

                    for i in range(3):
                        d_inputs["u_struct"][i :: self.struct_ndof] -= np.array(
                            d_in[i::3], dtype=float
                        )

                if "f_aero" in d_inputs:
                    # df_s/df_a^T psi = - dL/df_a^T * psi = -dD/du_s * psi
                    prod = np.zeros(self.aero_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydDduS(d_out, prod)
                    d_inputs["f_aero"] -= np.array(prod, dtype=float)

                if "x_aero0" in d_inputs:
                    # df_s/dx_a0^T * psi = - psi^T * dL/dx_a0 in F2F terminology
                    prod = np.zeros(self.aero_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydLdxA0(d_out, prod)
                    d_inputs["x_aero0"] -= np.array(prod, dtype=float)

                if "x_struct0" in d_inputs:
                    # df_s/dx_s0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
                    prod = np.zeros(self.struct_nnodes * 3, dtype=TransferScheme.dtype)
                    self.xfer.applydLdxS0(d_out, prod)
                    d_inputs["x_struct0"] -= np.array(prod, dtype=float)

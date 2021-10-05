#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import shutil
from funtofem import TransferScheme
from pyfuntofem.solver_interface import SolverInterface

import pysu2
try:
    import pysu2ad
except:
    pass

class SU2Interface(SolverInterface):
    """FUNtoFEM interface class for SU2."""

    def __init__(self, comm, model, su2_config, su2ad_config=None, qinf=1.0,
                 restart_file='restart_flow.dat',
                 solution_file='solution_flow.dat',
                 forward_options=None, adjoint_options=None):
        """Initialize the SU2 interface"""
        self.comm = comm
        self.qinf = qinf
        self.su2_config = su2_config
        self.su2ad_config = su2ad_config
        self.restart_file = restart_file
        self.solution_file = solution_file
        self.su2 = None
        self.su2ad = None

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        return

    def initialize(self, scenario, bodies, first_pass=False):
        # Instantiate the SU2 flow solver
        if first_pass or self.su2 is None:
            self.su2 = pysu2.CSinglezoneDriver(self.su2_config, 1, self.comm)

        # Keep track of what is a node on the surface
        self.surf_id = None
        self.num_local_surf_nodes = 0

        # Get the identifiers of the moving surface
        moving_marker_tags = self.su2.GetAllDeformMeshMarkersTag()
        if moving_marker_tags is None or len(moving_marker_tags) == 0:
            raise RuntimeError('No moving surface defined in the mesh')

        # Get the marker ids for each surface
        all_marker_ids = self.su2.GetAllBoundaryMarkers()
        self.surface_ids = []
        self.num_surf_nodes = []
        for tag in moving_marker_tags:
            if tag in all_marker_ids:
                self.surface_ids.append(all_marker_ids[tag])
                self.num_surf_nodes.append(self.su2.GetNumberVertices(self.surface_ids[-1]))

        # Keep track of the total number of surface nodes. Cast to an integer
        # since np.sum returns a zero float for an empty array
        self.num_total_surf_nodes = int(np.sum(self.num_surf_nodes))

        # Get the coordinates associated with the surface nodes
        bodies[0].aero_nnodes = self.num_total_surf_nodes
        bodies[0].aero_X = np.zeros(3*self.num_total_surf_nodes, dtype=TransferScheme.dtype)
        bodies[0].aero_loads = np.zeros(3*self.num_total_surf_nodes,
                                        dtype=TransferScheme.dtype)

        for ibody, body in enumerate(bodies):
            for index, surf_id in enumerate(self.surface_ids):
                offset = sum(self.num_surf_nodes[:index])

                for vert in range(self.num_surf_nodes[index]):
                    x, y, z = self.su2.GetInitialMeshCoord(surf_id, vert)
                    idx = 3*(vert + offset)
                    body.aero_X[idx] = x
                    body.aero_X[idx+1] = y
                    body.aero_X[idx+2] = z

        return 0

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of SU2.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
        #. Set the aerodynamic forces into the body data types

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        for ibody, body in enumerate(bodies):

            for index, surf_id in enumerate(self.surface_ids):
                offset = sum(self.num_surf_nodes[:index])

                if body.transfer is not None:
                    for vert in range(self.num_surf_nodes[index]):
                        idx = 3*(vert + offset)
                        u = body.aero_disps[idx]
                        v = body.aero_disps[idx+1]
                        w = body.aero_disps[idx+2]
                        self.su2.SetMeshDisplacement(surf_id, vert, u, v, w)

                if body.thermal_transfer is not None:
                    for vert in range(self.num_suf_nodes[index]):
                        idx = vert + offset
                        Twall = body.aero_temps[idx]
                        self.su2.SetVertexTemperature(surf_id, vert, Twall)

        # If this is an unsteady computation than we will need this:
        self.su2.ResetConvergence()
        self.su2.Preprocess(0)
        self.su2.Run()
        self.su2.Postprocess()
        self.su2.Monitor(0)
        self.su2.Output(0)

        # Pull out the forces from SU2
        for ibody, body in enumerate(bodies):
            if body.transfer:
                body.aero_loads[:] = 0.0
            if body.thermal_transfer:
                body.aero_heat_flux[:] = 0.0
                body.aero_heat_flux_mag[:] = 0.0

        for ibody, body in enumerate(bodies):
            for index, surf_id in enumerate(self.surface_ids):
                offset = sum(self.num_surf_nodes[:index])

                if body.transfer is not None:
                    for vert in range(self.num_surf_nodes[index]):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            fx, fy, fz = self.su2.GetFlowLoad(surf_id, vert)
                            body.aero_loads[3*vert] = self.qinf * fx
                            body.aero_loads[3*vert+1] = self.qinf * fy
                            body.aero_loads[3*vert+2] = self.qinf * fz

                if body.thermal_transfer is not None:
                    for vert in range(self.num_surf_nodes[index]):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            hx, hy, hz = self.su2.GetVertexHeatFluxes(surf_id, vert)
                            body.aero_heat_flux[3*vert] = hx
                            body.aero_heat_flux[3*vert+1] = hy
                            body.aero_heat_flux[3*vert+2] = hz

                            hmag = self.su2.GetVertexNormalHeatFlux(surf_id, vert)
                            body.aero_heat_flux_mag[vert] = hmag

        return 0

    def post(self,scenario, bodies, first_pass=False):
        if not first_pass and self.su2 is not None:
            self.su2.Postprocessing()

        # If this isn't the first pass, delete the flow object
        if not first_pass:
            del self.su2
            self.su2 = None

        # Copy the restart file to the solution file...
        self.comm.Barrier()
        if self.comm.rank == 0:
            shutil.copyfile(self.restart_file, self.solution_file)
        self.comm.Barrier()

        return

    def initialize_adjoint(self, scenario, bodies):
        # Instantiate the SU2 flow solver
        if self.su2ad is None:
            # Delete the primal if it exists at this point...
            if self.su2 is not None:
                del self.su2
                self.su2 = None

            # Create the discrete adjoint version of SU2
            self.su2ad = pysu2ad.CDiscAdjSinglezoneDriver(self.su2ad_config, 1, self.comm)

        # Keep track of what is a node on the surface
        self.surf_id = None
        self.num_local_surf_nodes = 0

        # Get the identifiers of the moving surface
        moving_marker_tags = self.su2ad.GetAllDeformMeshMarkersTag()
        if moving_marker_tags is None or len(moving_marker_tags) == 0:
            raise RuntimeError('No moving surface defined in the mesh')

        # Get the marker ids for each surface
        all_marker_ids = self.su2ad.GetAllBoundaryMarkers()
        self.surface_ids = []
        self.num_surf_nodes = []
        for tag in moving_marker_tags:
            if tag in all_marker_ids:
                self.surface_ids.append(all_marker_ids[tag])
                self.num_surf_nodes.append(self.su2ad.GetNumberVertices(self.surface_ids[-1]))

        # Keep track of the total number of surface nodes. Cast to an integer
        # since np.sum returns a zero float for an empty array
        self.num_total_surf_nodes = int(np.sum(self.num_surf_nodes))

        return

    def iterate_adjoint(self, scenario, bodies, step):

        func = 0

        for ibody, body in enumerate(bodies):
            # Extract the adjoint terms for the vertex forces

            for index, surf_id in enumerate(self.surface_ids):
                offset = sum(self.num_surf_nodes[:index])

                if body.transfer is not None:
                    psi_F = - body.dLdfa
                    print('psi_F = ', np.dot(psi_F.T, psi_F))
                    for vert in range(self.num_surf_nodes[index]):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            fx_adj = self.qinf * psi_F[3*vert, func]
                            fy_adj = self.qinf * psi_F[3*vert+1, func]
                            fz_adj = self.qinf * psi_F[3*vert+2, func]
                            self.su2ad.SetFlowLoad_Adjoint(surf_id, vert, fx_adj, fy_adj, fz_adj)

                if body.thermal_transfer is not None:
                    psi_Q_flux = - body.dQfluxdha
                    psi_Q_mag = - body.dQmagdta

                    for vert in range(self.num_surf_nodes[index]):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            # hx_adj = psi_Q_flux[3*vert, func]
                            # hy_adj = psi_Q_flux[3*vert+1, func]
                            # hz_adj = psi_Q_flux[3*vert+2, func]
                            # self.su2ad.SetVertexHeatFluxes_Adjoint(surf_id, vert,
                            #                                        hx_adj, hy_adj, hz_adj)

                            hmag_adj = psi_Q_mag[vert, func]
                            self.su2ad.SetVertexNormalHeatFlux_Adjoint(surf_id, vert, hmag_adj)

        self.su2ad.ResetConvergence()
        self.su2ad.Preprocess(0)
        self.su2ad.Run()
        self.su2ad.Postprocess()
        self.su2ad.Update()
        stopCalc = self.su2ad.Monitor(0)

        for ibody, body in enumerate(bodies):
            for index, surf_id in enumerate(self.surface_ids):
                offset = sum(self.num_surf_nodes[:index])

                if body.transfer is not None:
                    for vert in range(self.num_surf_nodes[index]):
                        idx = 3*(vert + offset)
                        u_adj, v_adj, w_adj = self.su2ad.GetMeshDisp_Sensitivity(surf_id, vert)
                        body.dGdua[idx, func] = u_adj
                        body.dGdua[idx+1, func] = v_adj
                        body.dGdua[idx+2, func] = w_adj

                if body.thermal_transfer is not None:
                    for vert in range(self.num_surf_nodes[index]):
                        idx = vert + offset
                        Twall_adj = self.su2ad.GetVertexTemperature_Adjoint(surf_id, vert)
                        body.dAdta[idx, func] = Twall_adj

            print('body.dGdua = ', np.dot(body.dGdua.T, body.dGdua))

        return 0

    def post_adjoint(self, scenario, bodies):
        if self.su2ad is not None:
            self.su2ad.Postprocessing()
            del self.su2ad
            self.su2ad = None

        return

    def set_functions(self, scenario, bodies):
        pass

    def set_variables(self, scenario, bodies):
        pass

    def get_functions(self, scenario, bodies):
        pass

    def get_function_gradients(self, scenario, bodies, offset):
        pass

    def get_coordinate_derivatives(self, scenario, bodies, step):
        pass

    def set_states(self, scenario, bodies, step):
        pass

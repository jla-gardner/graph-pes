Theory
======

An isolated atomic structure containing :math:`N` atoms is completely defined by:

* the positions of its atoms, :math:`\vec{R} \in \mathbb{R}^{N \times 3}`
* their atomic numbers, :math:`{Z} \in \mathbb{Z}^N`
* the unit cell, :math:`C \in \mathbb{R}^{3 \times 3}` (if the structure is periodic)

Energy
------

Since these three properties fully define the structure, it must be that the total energy, :math:`E \in \mathbb{R}`,
can be expressed solely as a function of these three properties:

.. math::

    E = f\left(\vec{R}, Z, C\right)


Forces
------

The force on atom :math:`i`, :math:`\vec{F}_i \in \mathbb{R}^3`, is given by the
negative gradient of the energy with respect to that atom's position:

.. math::

    \vec{F}_i = -\frac{\partial E}{\partial \vec{R}_i}

By using :class:`torch.Tensor` representations of :math:`\vec{R}`, :math:`Z`, and :math:`C`, and 
ensuring that the energy function, :math:`f`, makes use of ``torch`` operations, we can leverage
automatic differentiation, supplied by :func:`torch.autograd.grad`, to calculate the forces on the atoms in a structure
"for free" from any energy prediction.

Stress
------

Consider scaling a structure, that is "stretching" both the atomic positions and unit cell, by some amount,
:math:`1 + \lambda`, along the :math:`x` direction. This operation, :math:`\hat{O}_{\lambda}`, acts 
on the atomic positions, :math:`\vec{R}`, according to:

.. math::

    \hat{O}_{\lambda} \left(\vec{R}\right) = \vec{R} \times \begin{pmatrix}
        1 + \lambda & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1
    \end{pmatrix}  = \vec{R} + \lambda R_x \begin{pmatrix}
        1 \\
        0 \\
        0
    \end{pmatrix}

and analogously for the structure's unit cell, :math:`C`.
The response of the energy to this transformation gives an indication as to the stress acting on the structure. If 

.. math::
    \frac{\partial E\left[\hat{O}_{\lambda}(\vec{R}), Z, \hat{O}_{\lambda}(C)\right]}{\partial \lambda} \bigg|_{\lambda=0} =
    \frac{\partial E}{\partial \lambda} \bigg|_{\lambda=0} < 0 

then the energy decreases as the unit cell expands from its current state. 
This would indicate that the system is under *compressive* stress (along the :math:`x` direction) and "wants" to expand.

Now consider a more general scaling operation, :math:`\hat{\mathbf{O}}_{\mathbf{\lambda}}`, that symmetrically scales both the atomic positions and unit cell as:

.. math::

    \begin{aligned}
    \hat{\mathbf{O}}_{\mathbf{\lambda}} \left(\vec{R}\right) &= \vec{R} \times \begin{pmatrix}
        1 + \lambda_{xx} & \lambda_{xy} & \lambda_{xz} \\
        \lambda_{yx} & 1 + \lambda_{yy} & \lambda_{yz} \\
        \lambda_{zx} & \lambda_{zy} & 1 + \lambda_{zz}
    \end{pmatrix} \\
    &= \vec{R} + \vec{R} \times \begin{pmatrix}
        \lambda_{xx} & \lambda_{xy} & \lambda_{xz} \\
        \lambda_{yx} & \lambda_{yy} & \lambda_{yz} \\
        \lambda_{zx} & \lambda_{zy} & \lambda_{zz}
    \end{pmatrix}
    \end{aligned}

where, due to the symmetry of the expansion, :math:`\lambda_{ij} = \lambda_{ji} \quad \forall \; i \neq j \in \{x,y,z\}`.

The diagonal terms of this matrix again correspond to the compressive/dilative stress along each of the Cartesian axes.

The **off-diagonal terms** describe the shear stress, *i.e.* the tendency of the structure to slide in one plane relative to another.

In ``graph-pes``, we follow the common definition of the **stress tensor**, :math:`\mathbf{\sigma} \in \mathbb{R}^{3 \times 3}`, as the derivative
of the total energy with respect to these stretching coefficients, as normalised by the cell's volume, :math:`V = \det(\mathbf{C})`: [1]_

.. math::

    \mathbf{\sigma} = \frac{1}{V} \frac{\partial E}{\partial \mathbf{\lambda}} \bigg|_{\mathbf{\lambda} = 0} 
    \quad \quad 
    \sigma_{ij} = \frac{1}{V} \frac{\partial E}{\partial \lambda_{ij}} \bigg|_{\lambda_{ij} = 0}

We can again make use of automatic differentiation to calculate these stress tensors. To do this, we:

1. define a symmetrized :math:`\mathbf{\lambda} = 0^{3 \times 3}`. This is all zeros since we 
    - don't want to actually change the atomic positions or unit cell for the energy calculation
    - want to evaluate the derivative at :math:`\mathbf{\lambda} = 0`
2. apply the scaling operation, :math:`\hat{\mathbf{O}}_{\mathbf{\lambda}}`, to the atomic positions and unit cell
    - again this is a no-op due to evaluating the scaling operation at :math:`\mathbf{\lambda} = 0`, but introduces the scaling coefficients into the computational graph
3. evaluate the energy
4. calculate the derivative of the energy with respect to :math:`\mathbf{\lambda}` and normalise by the cell's volume

.. [1] F. Knuth et al. `All-electron formalism for total energy strain
   derivatives and stress tensor components for numeric atom-centered
   orbitals <https://www.sciencedirect.com/science/article/pii/S0010465515000090>`__.
   Computer Physics Communications 190, 33–50 (2015).

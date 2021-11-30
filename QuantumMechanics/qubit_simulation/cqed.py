"""
Fluxonium circuit object

Author: Haonan Xiong
based on python code by Konstantin Nesterov

Date: Nov. 23, 2021
"""

import numpy as np
import qutip as qt
import sys
sys.dont_write_bytecode = True

class Cavity(object):
    """A class representing fluxonium qubit"""

    def __init__(self, f, nlev=5):
        """The Hamiltonian is H = f a^dag a"""
        """Energies have unit of GHz"""
        self.f = f #Cavity frequency
        self.nlev = nlev #Number of energy levels simulated, nlevels < nhilbert
        self.type = 'cavity'

    def __str__(self):
        text = ('A cavity with f = {} '.format(self.f) + 'GHz.')
        return text

    @property
    def f(self):
        return self._f

    @property
    def nlev(self):
        return self._nlev

    @f.setter
    def f(self, value):
        if value <= 0:
            raise Exception('Cavity energy must be positive.')
        else:
            self._f = value
            self._reset_cache()

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('Number of levels must be positive.')
        else:
            self._nlev = value
            self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _a(self):
        """annihilation operator in the LC basis."""
        return qt.destroy(self.nlev)

    def _adag(self):
        """creation operator in the LC basis."""
        return qt.create(self.nlev)

    def _hamiltonian_lc(self):
        """Cavity Hamiltonian."""
        f = self.f
        a = self._a()
        adag = self._adag()
        return f * adag * a


    def _eigenspectrum_lc(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the LC basis.
        Hold values previously calculated, unless cache is reset"""
        if not eigvecs_flag:
            if self._eigvals is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals = H_lc.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals, self._eigvecs = H_lc.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None):
        """Eigenenergies of the cavity.

        Parameters
        ----------
        nlev : int, optional
            The number of cavity eigenstates if different from `self.nlev`.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        return self._eigenspectrum_lc()[0:nlev]

    def level(self, level_index):
        """Energy of a single level of the cavity.

        Parameters
        ----------
        level_ind : int
            The cavity level starting from zero.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_index < 0 or level_index >= self.nlev:
            raise Exception('The level is out of bounds')
        return self._eigenspectrum_lc()[level_index]

    def freq(self, level1, level2):
        """Transition energy/frequency between two levels of the cavity.

        Parameters
        ----------
        level1, level2 : int
            The cavity levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2` defined
            as the difference of energies. Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def H(self, nlev=None):
        """cavity Hamiltonian in its eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of cavity eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def one(self, nlev=None):
        """Identity operator in the cavity eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of cavity eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1:
            raise Exception('`nlev` is out of bounds.')
        return qt.qeye(nlev)

    def a(self, nlev=None):
        """Annihilation operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The annihilation operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        a_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                a_op[ind1, ind2] = self._a().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(a_op)


    def adag(self, nlev=None):
        """Creation operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The creation operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        adag_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                adag_op[ind1, ind2] = self._adag().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(adag_op)


    def a_ij(self, level1, level2):
        """The annihilation matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The cavity levels.

        Returns
        -------
        complex
            The matrix element of the annihilation operator.
        """
        if (level1 < 0 or level1 > self.nlev
                or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._a().matrix_element(
            evecs[level1].dag(), evecs[level2])

    def adag_ij(self, level1, level2):
        """The creation matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The cavity levels.

        Returns
        -------
        complex
            The matrix element of the creation operator.
        """
        if (level1 < 0 or level1 > self.nlev
                or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._adag().matrix_element(
            evecs[level1].dag(), evecs[level2])
    
    
import numpy as np
from ase.units import Pascal, _hplanck, kB, _e, _hbar, _amu
from pathlib import Path

boltzmann = kB # eV/kelvin
planck = _hbar  # eV*s
amu = _amu    # kg
def _random_rotation(pos, circlefrac = 1.0):
    # Translate to origin
    com = np.average(pos, axis=0)
    pos -= com

    randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums

    theta = theta * 2.0*circlefrac*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*circlefrac  # For magnitude of pole deflection.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    pos = np.einsum('ib,ab->ia', pos, M)
    return pos + com

def _random_translation(pos, rvecs):
    pos -= np.average(pos, axis=0)
    rnd = np.random.rand(3)
    new_cos = rnd[0]*rvecs[0] + rnd[1]*rvecs[1] + rnd[2]*rvecs[2]
    return pos + new_cos

def random_position(pos, rvecs):
    pos = _random_rotation(pos)
    pos = _random_translation(pos, rvecs)
    return pos

def vdw_overlap(atoms, vdw, n_frame, n_ads, select_ads):
    nat = len(atoms)
    pos, numbers = atoms.get_positions(), atoms.get_atomic_numbers()
    for i_ads in range(n_frame + n_ads*select_ads, n_frame + n_ads*(select_ads+1)):
        dists = atoms.get_distances(i_ads, np.arange(nat), mic=True)
        for i, d in enumerate(dists):
            if i >= n_frame + n_ads*select_ads and i < n_frame + n_ads*(select_ads+1):
                continue
            if d < vdw[numbers[i_ads]] + vdw[numbers[i]]:
                return True
    return False


class EOS(object):
    def __init__(self, mass=0.0):
        self.mass = mass

    def calculate_fugacity(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           f
                The fugacity
        """
        mu, Pref = self.calculate_mu_ex(T, P)
        fugacity = np.exp( mu/(boltzmann*T) )*Pref
        return fugacity

    def calculate_mu(self, T, P):
        """
           Evaluate the chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           mu
                The chemical potential
        """
        # Excess part
        mu, Pref = self.calculate_mu_ex(T,P)
        # Ideal gas contribution to chemical potential
        assert self.mass!=0.0
        lambd = 2.0*np.pi*self.mass*boltzmann*T/planck**2
        mu0 = -boltzmann*T*np.log( boltzmann*T/Pref*lambd**1.5)
        return mu0+mu

    def get_Pref(self, T, P0, deviation=1e-3):
        """
           Find a reference pressure at the given temperature for which the
           fluidum is nearly ideal.
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Optional arguments:**
           deviation
                When the compressibility factor Z deviates less than this from
                1, ideal gas behavior is assumed.
        """
        Pref = P0
        for i in range(100):
            rhoref = self.calculate_rho(T, Pref)
            Zref = Pref/rhoref/boltzmann/T
            # Z close to 1.0 means ideal gas behavior
            if np.abs(Zref-1.0)>deviation:
                Pref /= 2.0
            else: break
        if np.abs(Zref-1.0)>deviation:
            raise ValueError("Failed to find pressure where the fluidum is ideal-gas like, check input parameters")
        return Pref




class PREOS(EOS):
    """The Peng-Robinson equation of state"""
    def __init__(self, Tc, Pc, omega, mass=0.0, phase="vapour"):
        """
           The Peng-Robinson EOS gives a relation between pressure, volume, and
           temperature with parameters based on the critical pressure, critical
           temperature and acentric factor.
           **Arguments:**
           Tc
                The critical temperature of the species
           Pc
                The critical pressure of the species
           omega
                The acentric factor of the species
           **Optional arguments:**
           mass
                The mass of one molecule of the species. Some properties can be
                computed without this, so it is an optional argument
           phase
                Either "vapour" or "liquid". If both phases coexist at certain
                conditions, properties for the selected phase will be reported.
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.mass = mass
        self.phase = phase

        # Some parameters derived from the input parameters
        self.a = 0.457235 * self.Tc**2 / self.Pc
        self.b = 0.0777961 * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2

    @classmethod
    def from_name(cls, compound):
        """
           Initialize a Peng-Robinson EOS based on the name of the compound.
           Only works if the given compound name is included in
           'critical_acentric.csv'
        """
        # Read the data file containing parameters for a number of selected compounds
        module_dir = Path(__file__).parent
        fn = module_dir / 'critical_acentric.csv'
#        fn = pkg_resources.resource_filename(yaff.__name__, 'data/critical_acentric.csv')
        dtype=[('compound','S20'),('mass','f8'),('Tc','f8'),('Pc','f8'),('omega','f8'),]
        data = np.genfromtxt(fn, dtype=dtype, delimiter=',')
        # Select requested compound
        if not compound.encode('utf-8') in data['compound']:
            raise ValueError("Could not find data for %s in file %s"%(compound,fn))
        index = np.where( compound.encode('utf-8') == data['compound'] )[0]
        assert index.shape[0]==1
        mass = data['mass'][index[0]] # amu
        Tc = data['Tc'][index[0]] # K
        Pc = data['Pc'][index[0]]*1e6*Pascal # eV/A3
        omega = data['omega'][index[0]]
        return cls(Tc, Pc, omega, mass=mass)

    def set_conditions(self, T, P):
        """
           Set the parameters that depend on T and P
           **Arguments:**
           T
                Temperature
           P
                Pressure
        """
        self.Tr = T / self.Tc  # reduced temperature
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.Tr)))**2
        self.A = self.a * self.alpha * P / T**2
        self.B = self.b * P / T

    def polynomial(self, Z):
        """
           Evaluate the polynomial form of the Peng-Robinson equation of state
           If returns zero, the point lies on the PR EOS curve
           **Arguments:**
           Z
                Compressibility factor
        """
        return Z**3 - (1 - self.B) * Z**2 + (self.A - 2*self.B - 3*self.B**2) * Z - (
                self.A * self.B - self.B**2 - self.B**3)

    def polynomial_roots(self):
        """
            Find the real roots of the polynomial form of the Peng-Robinson
            equation of state
        """
        a = - (1 - self.B)
        b = self.A - 2*self.B - 3*self.B**2
        c = - (self.A * self.B - self.B**2 - self.B**3)
        Q = (a**2-3*b)/9
        R = (2*a**3-9*a*b+27*c)/54
        M = R**2-Q**3
        if M>0:
            S = np.cbrt(-R+np.sqrt(M))
            T = np.cbrt(-R-np.sqrt(M))
            Z = S+T-a/3

        else:
            theta = np.arccos(R/np.sqrt(Q**3))
            x1 = -2.0*np.sqrt(Q)*np.cos(theta/3)-a/3
            x2 = -2.0*np.sqrt(Q)*np.cos((theta+2*np.pi)/3)-a/3
            x3 = -2.0*np.sqrt(Q)*np.cos((theta-2*np.pi)/3)-a/3
            solutions = np.array([x1,x2,x3])
            solutions = solutions[solutions>0.0]
            if self.phase=='vapour':
                Z = np.amax(solutions)
            elif self.phase=='liquid':
                Z = np.amin(solutions)
            else: raise NotImplementedError
        return Z

    def calculate_rho(self, T, P):
        """
           Calculate the particle density at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           rho
                The particle density
        """
        self.set_conditions(T, P)
        Z = self.polynomial_roots()
        return P/Z/boltzmann/T

    def calculate_mu_ex(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           mu
                The excess chemical potential
           Pref
                The pressure at which the reference chemical potential was calculated
        """
        # Find a reference pressure at the given temperature for which the fluidum
        # is nearly ideal
        Pref = self.get_Pref(T, P)
        # Find compressibility factor using rho
        rho = self.calculate_rho(T, P)
        Z = P/rho/boltzmann/T
        # Add contributions to chemical potential at requested pressure
        mu = Z - 1 - np.log(Z - self.B) - self.A / np.sqrt(8) / self.B * np.log(
                    (Z + (1 + np.sqrt(2)) * self.B) / (Z + (1 - np.sqrt(2)) * self.B))
        mu += np.log(P/Pref)
        mu *= T*boltzmann
        return mu, Pref



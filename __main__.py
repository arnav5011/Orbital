import numpy as np
from math import factorial
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider

def get_quantum_numbers():
    while True:
        try:
            n = int(input("Enter Principal Quantum Number (n): "))
            if n > 7:
                raise ValueError("Principal Quantum Number (n) must be 7 or less.")

            l = int(input("Enter Angular Momentum Quantum Number (l): "))
            if l > 5:
                raise ValueError("Angular Momentum Quantum Number (l) must be 4 or less.")
            if l >= n:
                raise ValueError("Angular Momentum Quantum Number (l) must be less than Principal Quantum Number (n).")

            m = int(input("Enter Magnetic Quantum Number (m): "))
            if not (-l <= m <= l):
                raise ValueError(f"Magnetic Quantum Number (m) must be between {-l} and {l}.")
            return n, l, m

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def associated_laguerre_polynomials(n, l, r):
    sum = 0
    for i in range(0, n - l):
        x = (-1) ** i
        y = (factorial(n + l))/(factorial(n - l - 1 - i) * factorial(2 * l + 1 + i))
        z = ((2 * r) ** i) / factorial(i)
        sum += x * y * z
    return sum

def generalized_binomial(alpha, k):
    product = 1
    for i in range(1, k+1):
        product = product * (alpha - k + i)/(i)
    return product

def associated_legendre_polynomial(m, l, x):
    pre_factor = (-1)**m * (1-x*x)**(m/2) * 2**l #(-1)^m * (1-x^2)^(m/2) * 2^l
    sum = 0
    for k in range(m,l+1):
        a = factorial(k)/factorial(k-m)
        b = generalized_binomial(l,k)
        mix = (l+k-1)/2
        c = generalized_binomial(mix,l)
        d = x**(k-m)
        sum = sum + a*b*c*d
    return pre_factor * sum

def get_cubic_harmonic(spherical, m):
    if m>0:
        return 1/np.sqrt(2)* (spherical + np.conj(spherical))
    elif m == 0:
        return spherical
    if m<0:
        return 1j/np.sqrt(2) * (spherical - np.conj(spherical))

def get_radial_probability_density_values(n, l):
    a0 = 5.29177210544e-11  # Bohr radius in meters
    
    normalization = np.sqrt(((2 / (n * a0)) ** 3) * (factorial(n - l - 1) / (2 * n * (factorial(n + l)))))
    
    array_r = np.linspace(0, n * n * 5 * a0, num=n*n*20)
    
    array_R = np.zeros_like(array_r) 
    array_pdf = np.zeros_like(array_r)
    
    for i in range(len(array_r)):
        r_scaled = array_r[i] / a0
        middle = (2 * r_scaled/n) ** l * np.exp(-r_scaled/n)
        
        array_R[i] = normalization * middle * associated_laguerre_polynomials(n, l, r_scaled/n)
        
        array_pdf[i] = array_R[i] ** 2 * array_r[i] ** 2 
    
    return array_r, array_R, array_pdf

def get_spherical_harmonics(l,m):
    t = abs(m)
    normalization = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-t)/factorial(l+t))
    theta_array = np.linspace(0, np.pi, 200)
    phi_array = np.linspace(0, 2*np.pi, 200)
    harmonics_array = np.ones(len(theta_array) * len(phi_array), dtype = complex)
    for i in range(len(phi_array)):
        for j in range(len(theta_array)):
            a = np.exp(1j*t*phi_array[i])
            legendre = associated_legendre_polynomial(t, l, np.cos(theta_array[j]))
            harmonics_array[len(theta_array)*i + j] = normalization * a * legendre
    if m<0: harmonics_array = np.conj(harmonics_array)
    
    return theta_array, phi_array, harmonics_array
           
def plot_radial_function(array_r, array_R, n):
    a0 = 5.29177210544e-11
    fig = plt.figure()
    plt.xlabel(f"r/a0")
    plt.ylabel("Radial Function (R)")
    plt.plot(array_r/(a0),array_R)
    plt.show()

def plot_radial_density(array_r, array_pdf, n):
    a0 = 5.29177210544e-11
    fig = plt.figure()
    plt.xlabel(f"r/a0")
    plt.ylabel("Radial PDF = r^2 * R^2")
    plt.plot(array_r/(n*a0),array_pdf)
    plt.show()

def plot_harmonics(theta_array, phi_array, harmonics_array, l, m):
    x_array = np.zeros_like(harmonics_array, dtype=np.float64)
    y_array = np.zeros_like(harmonics_array, dtype=np.float64)
    z_array = np.zeros_like(harmonics_array, dtype=np.float64)

    # Loop through phi and theta arrays to compute Cartesian coordinates
    for i in range(len(phi_array)):
        for j in range(len(theta_array)):
            idx = len(theta_array) * i + j
            r = np.abs(harmonics_array[idx])  # Use the magnitude of spherical harmonics for r

            # Spherical to Cartesian conversion
            x_array[idx] = r * np.sin(theta_array[j]) * np.cos(phi_array[i])
            y_array[idx] = r * np.sin(theta_array[j]) * np.sin(phi_array[i])
            z_array[idx] = r * np.cos(theta_array[j])

    # Get magnitude for color mapping
    magnitude = np.abs(harmonics_array)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color mapping based on the magnitude of harmonics
    scatter = ax.scatter(x_array, y_array, z_array, c=magnitude, cmap=cm.viridis, s=5, alpha=0.7)
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    
    slider_z = Slider(ax_slider, 'Z Stretch', 1, 5, valinit=1, valstep=0.1)

    # Set plot title and labels
    ax.set_title(f'Harmonics: l={l}, m={m}', fontsize=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

def get_orbital_name(n,l):
    orbit = f"{n}"
    if l == 0:
        orbit += "s"
    elif l == 1:
        orbit += "p"
    elif l == 2:
        orbit += "d"
    else:
        orbit += "f"
    return orbit
    
def main():
    n,l,m = get_quantum_numbers()
    array_r, array_R, array_pdf = get_radial_probability_density_values(n,l)
    plot_radial_function(array_r, array_R, n)
    plot_radial_density(array_r, array_pdf, n)
    theta, phi, y = get_spherical_harmonics(l,m)
    plot_harmonics(theta, phi, y, l, m)
    cubic_harmoncis = get_cubic_harmonic(y,m)
    plot_harmonics(theta, phi, cubic_harmoncis, l, m)

if __name__ == "__main__":
    main()
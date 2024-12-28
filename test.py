import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib import cm

# Associated Laguerre Polynomials function (original)
def associated_laguerre_polynomials(n, l, r):
    sum = 0
    for i in range(0, n - l):
        x = (-1) ** i
        y = (factorial(n + l)) / (factorial(n - l - 1 - i) * factorial(2 * l + 1 + i))
        z = ((2 * r) ** i) / factorial(i)
        sum += x * y * z
    return sum

# Generalized Binomial function
def generalized_binomial(alpha, k):
    product = 1
    for i in range(1, k+1):
        product = product * (alpha - k + i) / (i)
    return product

# Associated Legendre Polynomial function (original)
def associated_legendre_polynomial(m, l, x):
    pre_factor = (-1)**m * (1 - x*x)**(m/2) * 2**l
    sum = 0
    for k in range(m, l+1):
        a = factorial(k) / factorial(k - m)
        b = generalized_binomial(l, k)
        mix = (l + k - 1) / 2
        c = generalized_binomial(mix, l)
        d = x**(k - m)
        sum += a * b * c * d
    return pre_factor * sum

# Radial probability density values function (original)
def get_radial_probability_density_values(n, l):
    a0 = 5.29177210544e-11  # Bohr radius in meters
    
    normalization = np.sqrt(((2 / (n * a0)) ** 3) * (factorial(n - l - 1) / (2 * n * (factorial(n + l) ** 3))))
    
    array_r = np.linspace(0, n * n * 5 * a0, num=n*n*20)
    array_R = np.zeros_like(array_r)
    array_pdf = np.zeros_like(array_r)
    
    for i in range(len(array_r)):
        r_scaled = array_r[i] / a0
        middle = (2 * r_scaled / n) ** l * np.exp(-r_scaled / n)
        array_R[i] = normalization * middle * associated_laguerre_polynomials(n, l, r_scaled / n)
        array_pdf[i] = array_R[i] ** 2 * array_r[i] ** 2 
    
    return array_r, array_R, array_pdf

# Spherical Harmonics function (adjusted to 1D)
def get_spherical_harmonics(l, m):
    m = abs(m)
    normalization = np.sqrt((2*l+1) / (4*np.pi) * factorial(l - m) / factorial(l + m))
    
    theta_array = np.linspace(0, np.pi, 100)
    phi_array = np.linspace(0, 2 * np.pi, 200)
    
    harmonics_array = np.zeros(len(theta_array) * len(phi_array), dtype=complex)
    
    # Flatten the 2D loop into a 1D array
    for i in range(len(phi_array)):
        for j in range(len(theta_array)):
            idx = len(theta_array) * i + j
            a = np.exp(1j * m * phi_array[i])
            legendre = associated_legendre_polynomial(m, l, np.cos(theta_array[j]))
            harmonics_array[idx] = normalization * a * legendre
            
    return theta_array, phi_array, harmonics_array

# Wave Function Calculation (with 1D harmonics_array)
def get_wave_function(n, l, m):
    r_array, R_array, R_pdf_array = get_radial_probability_density_values(n, l)
    theta_array, phi_array, harmonics_array = get_spherical_harmonics(l, m)
    
    # Initialize wave function as 3D array to match the size of r, theta, phi combinations
    wave_function_pdf = np.zeros((len(r_array), len(theta_array), len(phi_array)), dtype=float)
    
    # Calculate the wave function and its PDF at every (r, theta, phi) point
    for i, r in enumerate(r_array):
        for j, theta in enumerate(theta_array):
            for k, phi in enumerate(phi_array):
                # Find index for spherical harmonics (flattened)
                harmonics_idx = len(theta_array) * k + j
                radial_part = R_array[i]
                angular_part = harmonics_array[harmonics_idx]
                
                # Wave function value (radial * spherical harmonics)
                wave_function_value = radial_part * angular_part
                
                # PDF is |psi|^2
                wave_function_pdf[i, j, k] = np.abs(wave_function_value) ** 2
    
    return r_array, theta_array, phi_array, wave_function_pdf

# Plot Wave Function PDF in 3D (new)
def plot_wave_function_pdf(theta_array, phi_array, r_array, wave_function_pdf):
    """
    Function to plot the probability density function (PDF) of a wave function in spherical coordinates.
    
    Parameters:
    - theta_array: Array of theta values (angles from z-axis).
    - phi_array: Array of phi values (angles in the xy-plane).
    - r_array: Array of radial distances at which the wave function is evaluated.
    - wave_function_pdf: The probability density function (|ψ(r, θ, φ)|²) values.
    """
    # Create grids for theta and phi
    theta_grid, phi_grid = np.meshgrid(theta_array, phi_array)
    
    # Initialize Cartesian coordinate arrays
    x_grid = np.zeros_like(wave_function_pdf)
    y_grid = np.zeros_like(wave_function_pdf)
    z_grid = np.zeros_like(wave_function_pdf)
    
    # Loop through the PDF values and compute Cartesian coordinates
    for i in range(len(r_array)):
        # Use sqrt of the PDF to get the amplitude (|ψ|, rather than |ψ|²)
        r_component = np.sqrt(wave_function_pdf[i])  # Magnitude of the wave function
        
        # Spherical to Cartesian transformation
        x_grid[i] = r_component * np.sin(theta_grid[i]) * np.cos(phi_grid[i])
        y_grid[i] = r_component * np.sin(theta_grid[i]) * np.sin(phi_grid[i])
        z_grid[i] = r_component * np.cos(theta_grid[i])
    
    # Get magnitude for color mapping
    magnitude = np.abs(np.sqrt(wave_function_pdf))
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with color mapping based on the magnitude of the wave function
    scatter = ax.scatter(x_grid, y_grid, z_grid, c=magnitude, cmap=cm.viridis, s=5, alpha=0.7)
    
    # Add color bar to indicate magnitude
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Wave Function Magnitude')
    
    # Set plot title and labels
    ax.set_title('Wave Function PDF in 3D', fontsize=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Display the plot
    plt.show()
# Example usage
n, l, m = 3, 2, 1
r_array, theta_array, phi_array, wave_function_pdf = get_wave_function(n, l, m)
plot_wave_function_pdf(r_array, theta_array, phi_array, wave_function_pdf)

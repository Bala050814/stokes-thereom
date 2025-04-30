import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Stokes' Theorem Calculator", page_icon="🧮")

# Title
st.title("🧮 Stokes' Theorem Calculator")

st.write(r"""
This app calculates the **line integral** of a vector field \( \vec{F} \) over a curve \( C \) 
according to **Stokes' Theorem**, and shows a **3D graphical representation**!
""")

# Define symbols
x, y, z, t = sp.symbols('x y z t')

# Section: Vector Field
st.header("1️⃣ Define the Vector Field \( \vec{F}(x, y, z) \)")
F1 = st.text_input("F₁(x, y, z) (i-component)", value="y")
F2 = st.text_input("F₂(x, y, z) (j-component)", value="-x")
F3 = st.text_input("F₃(x, y, z) (k-component)", value="0")

# Section: Curve
st.header("2️⃣ Define the Parameterized Curve \( \vec{r}(t) \)")
r1 = st.text_input("x(t)", value="cos(t)")
r2 = st.text_input("y(t)", value="sin(t)")
r3 = st.text_input("z(t)", value="0")

# Section: t limits
st.header("3️⃣ Limits for Parameter t")
t_min = st.number_input("Start t:", value=0.0)
t_max = st.number_input("End t:", value=float(2 * sp.pi.evalf()), format="%.5f")

# --- Diagram Section ---
st.header("📈 3D Visual Representation of Surface and Curve")

# 3D Plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a circular surface (disk in xy-plane)
theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, 1, 50)
theta, r = np.meshgrid(theta, r)
X = r * np.cos(theta)
Y = r * np.sin(theta)
Z = np.zeros_like(X)

# Plot surface
ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', edgecolor='grey')

# Boundary curve (circle)
t_vals = np.linspace(0, 2 * np.pi, 100)
x_curve = np.cos(t_vals)
y_curve = np.sin(t_vals)
z_curve = np.zeros_like(t_vals)
ax.plot3D(x_curve, y_curve, z_curve, 'r', linewidth=3, label="Boundary Curve C")

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Surface S with Boundary Curve C")
ax.legend()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 0.5])

# Display 3D plot in Streamlit
st.pyplot(fig)

st.write(r"""
**Stokes' theorem** relates a surface integral of the curl of a vector field over a surface \( S \) 
to a line integral of the vector field over its boundary curve \( C \):

\[
\oint_C \vec{F} \cdot d\vec{r} = \iint_S (\nabla \times \vec{F}) \cdot d\vec{S}
\]
""")

# Button to calculate line integral
if st.button("🔵 Calculate Line Integral"):

    try:
        # Parse the field and curve components
        Fx = sp.sympify(F1, locals={"x": x, "y": y, "z": z})
        Fy = sp.sympify(F2, locals={"x": x, "y": y, "z": z})
        Fz = sp.sympify(F3, locals={"x": x, "y": y, "z": z})

        rx = sp.sympify(r1, locals={"t": t})
        ry = sp.sympify(r2, locals={"t": t})
        rz = sp.sympify(r3, locals={"t": t})

        F = sp.Matrix([Fx, Fy, Fz])
        r_vec = sp.Matrix([rx, ry, rz])

        # Compute dr/dt
        dr_dt = r_vec.diff(t)

        # Substitute r(t) into F
        F_sub = F.subs({x: rx, y: ry, z: rz})

        # Dot product F(r(t)) ⋅ dr/dt
        integrand = F_sub.dot(dr_dt)

        # Perform the definite integral
        line_integral = sp.integrate(integrand, (t, t_min, t_max))

        # Display results
        st.success("✅ Line integral computed successfully!")
        st.latex(r"\oint_C \vec{F} \cdot d\vec{r} = " + sp.latex(line_integral))

        with st.expander("🔵 Show Intermediate Computations"):
            st.latex(r"\vec{r}(t) = " + sp.latex(r_vec))
            st.latex(r"\frac{d\vec{r}}{dt} = " + sp.latex(dr_dt))
            st.latex(r"\vec{F}(\vec{r}(t)) = " + sp.latex(F_sub))
            st.latex(r"Integrand = " + sp.latex(integrand))

    except Exception as e:
        st.error(f"❌ Error during computation: {e}")

# Option: Compute and Show Curl
if st.checkbox("Show Curl of \\( \\vec{F} \\) and visualize vectors"):

    try:
        # Vector Field
        F = sp.Matrix([
            sp.sympify(F1, locals={"x": x, "y": y, "z": z}),
            sp.sympify(F2, locals={"x": x, "y": y, "z": z}),
            sp.sympify(F3, locals={"x": x, "y": y, "z": z})
        ])

        # Calculate Curl
        curl_F = sp.Matrix([
            sp.diff(F[2], y) - sp.diff(F[1], z),
            sp.diff(F[0], z) - sp.diff(F[2], x),
            sp.diff(F[1], x) - sp.diff(F[0], y)
        ])

        st.subheader("🔵 Curl of the Field:")
        st.latex(r"\nabla \times \vec{F} = " + sp.latex(curl_F))

        # 3D Vector Field Visualization
        fig2 = plt.figure(figsize=(7, 7))
        ax2 = fig2.add_subplot(111, projection='3d')

        # Sample grid points (in XY plane, z=0)
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        grid_z = np.zeros_like(grid_x)

        # Evaluate curl at grid points
        curl_func = sp.lambdify((x, y, z), curl_F, "numpy")
        curl_vectors = curl_func(grid_x, grid_y, grid_z)

        u = curl_vectors[0]
        v = curl_vectors[1]
        w = curl_vectors[2]

        # Plot quiver (arrows for curl vectors)
        ax2.quiver(grid_x, grid_y, grid_z, u, v, w, length=0.2, normalize=True, color='green')

        # Surface and Boundary again
        ax2.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='grey')
        ax2.plot3D(x_curve, y_curve, z_curve, 'r', linewidth=3, label="Boundary Curve C")

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title("Visualization of Curl Vectors over Surface")
        ax2.set_box_aspect([1, 1, 0.5])

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error computing or plotting curl: {e}")

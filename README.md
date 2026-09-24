# Direction of Arrival Estimation Using MUSIC

## About This Project

I am building this project to understand how Direction of Arrival (DOA) estimation works using antenna arrays and signal processing.

The main algorithm used in this project is MUSIC (Multiple Signal Classification). I started from the basic signal model and built the project step by step instead of using a ready-made DOA implementation.

The current version uses a Uniform Linear Array (ULA) and estimates the direction of multiple incoming signals from the spatial information contained in the signals received by the antenna elements.

I am mainly using Python for the simulation and signal processing.

---

# 1. What is Direction of Arrival?

Direction of Arrival means finding the direction from which a signal is arriving at an antenna array.

For example, imagine a signal arriving at an antenna array from an angle theta:

```text

                     Incoming signal

                           \\

                            \\

                             \\  theta

                              \\

                               \\

       o----o----o----o----o----o----o----o

      A0   A1   A2   A3   A4   A5   A6   A7

                  Antenna Array

```

The objective is to look at the signals received by all the antennas and estimate the angle without directly giving the algorithm the answer.

DOA estimation is useful in areas such as radar, wireless communications, radio direction finding, sonar, remote sensing and sensor arrays.

In this project I am starting with a simple 1D case, where the antenna elements are arranged in a straight line.

---

# 2. What is a ULA?

ULA means Uniform Linear Array.

It is a set of antenna elements placed in a straight line with equal spacing.

For this project I am using 8 antenna elements:

```text

A0     A1     A2     A3     A4     A5     A6     A7

o------o------o------o------o------o------o------o

        Uniform Linear Array (ULA)

```

The antenna spacing is:

```math

d = \frac{\lambda}{2}

```

where:

- `d` = distance between two adjacent antenna elements

- `lambda` = wavelength of the signal

I use half-wavelength spacing in the simulation because it is a common array configuration and helps avoid spatial aliasing over the normal scan range.

---

# 3. Why can an antenna array find the direction?

When a plane wave reaches an antenna array at an angle, it does not reach every antenna with exactly the same phase.

There is a phase difference between adjacent elements.

For a ULA, the phase difference can be written as:

```math

\Delta\phi = -2\pi\frac{d}{\lambda}\sin(\theta)

```

For example, in one of my experiments:

```math

\theta = 30^\circ

```

and:

```math

d = \frac{\lambda}{2}

```

Therefore:

```math

\Delta\phi = -2\pi\left(\frac{1}{2}\right)\sin(30^\circ) = -90^\circ

```

So the antennas receive the same underlying signal with a different phase progression.

This phase difference contains information about the direction of the incoming signal. This is the basic idea behind the DOA estimation used in this project.

---

# 4. Steering Vector

The steering vector describes the phase relationship across all the antenna elements for a particular direction.

For a ULA with `M` elements:

$$
\mathbf{a}(\theta)=
\begin{bmatrix}
1 \\
e^{-j2\pi\frac{d}{\lambda}\sin(\theta)} \\
e^{-j2\pi\frac{2d}{\lambda}\sin(\theta)} \\
\vdots \\
e^{-j2\pi\frac{(M-1)d}{\lambda}\sin(\theta)}
\end{bmatrix}
$$

    
In the Python code, I generate this vector in:

```text

src/array_model.py

```

The first experiment was only about understanding this steering vector before moving to the complete MUSIC algorithm.

---

# 5. Signal Model

The first step was to generate a complex baseband source signal.

For a single source, the received array data can be represented as:

```math

X = a(\theta)s(t)

```

For multiple sources, the model becomes:

```math

X = AS

```

where:

- `X` = received array data

- `A` = steering matrix

- `S` = source signals

When noise is added, the model becomes:

```math

X = AS + N

```

where `N` represents noise.

For the simulation I use complex AWGN (Additive White Gaussian Noise).

The signal model and noise generation are implemented in:

```text

src/signal_generation.py

```

---

# 6. Why is Noise Important?

An ideal signal is not enough to test a radar or signal-processing algorithm.

Real receivers contain noise, so I added AWGN to the received array data.

I tested different SNR values such as:

```text

20 dB

10 dB

 0 dB

-10 dB

```

At high SNR, the signal structure is easier to observe.

At low SNR, the noise becomes stronger and the phase information becomes harder to see from an individual sample.

This is important because MUSIC is not supposed to work only with a perfect signal. It has to use the spatial information present in many noisy measurements.

---

# 7. Covariance Matrix

MUSIC does not estimate the direction from only one sample.

The antenna array collects many snapshots of the received signal.

In the current experiment I use:

```text

8 antenna elements

1000 snapshots

```

So the received data matrix has the shape:

```math

X \in \mathbb{C}^{8\times1000}

```

From this data I calculate the spatial covariance matrix:

```math

R_{xx} = \frac{1}{N}XX^H

```

where:

- `N` = number of snapshots

- `X^H` = conjugate transpose of `X`

- `Rxx` = spatial covariance matrix

The covariance matrix has the shape:

```math

R_{xx} \in \mathbb{C}^{8\times8}

```

The covariance matrix contains information about the spatial relationship between the antenna elements.

I calculate this in:

```text

src/covariance.py

```

---

# 8. Eigenvalue Decomposition

The next step is to perform eigenvalue decomposition on the covariance matrix.

I use `numpy.linalg.eigh()` because the covariance matrix is Hermitian.

The decomposition is:

```math

R_{xx} = E\Lambda E^H

```

where:

- `E` contains the eigenvectors

- `Lambda` contains the eigenvalues

For one signal source and 8 antenna elements, I expect:

```text

1 signal dimension

7 noise dimensions

```

For two independent signal sources:

```text

2 signal dimensions

6 noise dimensions

```

For example, in the two-source experiment I obtained eigenvalues approximately like:

```text

0.1819

0.1929

0.2010

0.2078

0.2143

0.2327

7.1321

9.1621

```

The two larger eigenvalues are associated with the two signal dimensions, while the remaining eigenvalues form the noise part.

The eigenvalue and eigenvector calculation is implemented in:

```text

src/eigendecomposition.py

```

---

# 9. Signal Subspace and Noise Subspace

This is the main idea behind MUSIC.

The eigenvectors can be separated into signal and noise subspaces.

For two sources and eight antennas:

```text

8 eigenvectors

     |

     +---- 2 signal eigenvectors

     |

     +---- 6 noise eigenvectors

```

The MUSIC algorithm uses the noise subspace.

At the correct direction, the steering vector is approximately orthogonal to the noise subspace.

This can be written as:

```math

a^H(\theta)E_n \approx 0

```

where:

- `a(theta)` = steering vector

- `En` = noise-subspace eigenvectors

This property is what allows MUSIC to estimate the DOA.

---

# 10. MUSIC Pseudospectrum

The MUSIC pseudospectrum used in this project is:

```math

P_{MUSIC}(\theta)=\frac{1}{\left|a^H(\theta)E_nE_n^Ha(\theta)\right|}

```

The algorithm scans a range of angles.

In the current experiment I scan:

```text

-90 degrees to +90 degrees

```

For every angle, the steering vector is calculated and compared with the noise subspace.

At the true signal direction, the denominator becomes very small. This creates a large peak in the MUSIC spectrum.

The basic process is:

```text

Scan angle
    |
    v
Generate steering vector
    |
    v
Compare with noise subspace
    |
    v
Calculate MUSIC value
    |
    v
Move to next angle
    |
    v
Find peaks

```

The implementation is in:

```text

src/music.py

```

---

# 11. Single-Source Experiment

I first tested MUSIC with one signal source.

The source was placed at:

```math

\theta = 30^\circ

```

The main parameters were:

```text

Number of antennas : 8

Element spacing    : 0.5 lambda

Snapshots          : 1000

SNR                : 10 dB

True DOA           : 30 degrees

```

The estimated DOA was:

```text

30.0 degrees

```

The resulting MUSIC spectrum shows a clear peak around 30 degrees.

![Single-source MUSIC spectrum](results/music_spectrum_one_source.png)

---

# 12. Two-Source Experiment

After testing one source, I extended the simulation to two independent sources.

The two true directions were:

```math

\theta_1 = -20^\circ

```

and:

```math

\theta_2 = 35^\circ

```

The parameters were:

```text

Number of antennas : 8

Element spacing    : 0.5 lambda

Snapshots          : 1000

SNR                : 10 dB

Number of sources  : 2

True DOAs           : -20 degrees, 35 degrees

```

The MUSIC algorithm estimated:

```text

Source 1:

True DOA      = -20.0 degrees

Estimated DOA = -20.0 degrees

Error         = 0.0 degrees

Source 2:

True DOA      = 35.0 degrees

Estimated DOA = 35.1 degrees

Error         = 0.1 degrees

```

This shows two separate peaks in the MUSIC spectrum.

![Two-source MUSIC spectrum](results/music_spectrum_two_sources.png)

---

# 13. DOA Estimation Error

I calculate the estimation error by comparing the estimated direction with the known direction used in the simulation.

```math

e = \hat{\theta} - \theta_{true}

```

For the current two-source experiment:

```math

e_1 = -20.0^\circ - (-20.0^\circ) = 0.0^\circ

```

and:

```math

e_2 = 35.1^\circ - 35.0^\circ = 0.1^\circ

```

The known direction is available here because this is a controlled simulation. In a real measurement, the true DOA would not be known in advance.

---

# 14. Complete Processing Flow

The complete processing chain in the current version is:

```text

Source Signals
      |
      v
ULA Steering Vectors
      |
      v
Received Array Data
      |
      v
Add AWGN
      |
      v
Spatial Covariance Matrix
      |
      v
Eigenvalue Decomposition
      |
      v
Signal / Noise Subspaces
      |
      v
MUSIC Pseudospectrum
      |
      v
Peak Detection
      |
      v
Estimated DOAs

```

In mathematical form:

```math

S \rightarrow A \rightarrow X = AS + N \rightarrow R_{xx} \rightarrow E,\Lambda \rightarrow E_n \rightarrow P_{MUSIC}(\theta) \rightarrow \hat{\theta}

```

---

# 15. Project Structure

The current project is organized as:

```text

doa-estimation-music/

|

|-- .gitignore

|-- README.md

|-- main.py

|

|-- results/

|   |-- music_spectrum.png

|   |-- music_spectrum_two_sources.png

|

|-- src/

    |-- array_model.py

    |-- signal_generation.py

    |-- covariance.py

    |-- eigendecomposition.py

    |-- music.py

```

Each file has a separate purpose.

### `main.py`

Runs the complete experiment.

### `array_model.py`

Generates the ULA steering vector.

### `signal_generation.py`

Generates source signals and received array data, including AWGN.

### `covariance.py`

Calculates the spatial covariance matrix.

### `eigendecomposition.py`

Calculates the eigenvalues and eigenvectors.

### `music.py`

Calculates the MUSIC pseudospectrum and estimates the DOAs.

### `results/`

Stores the generated experiment results.

---

# 16. Technologies Used

- Python 3

- NumPy

- Matplotlib

- Git

- GitHub

---

# 17. Running the Project

Clone the repository:

```bash

git clone https://github.com/RamaRajuY/doa-estimation-music.git

```

Go into the project directory:

```bash

cd doa-estimation-music

```

Install the required Python packages:

```bash

python -m pip install numpy matplotlib

```

Run the experiment:

```bash

python main.py

```

The program generates the MUSIC spectrum and saves the result in the `results` folder.

---

# 18. What I Want to Add Next

This project is still a work in progress.

The next experiments I want to add are:

1. DOA estimation performance versus SNR

2. DOA estimation error analysis

3. Closely spaced sources

4. Effect of the number of antenna elements

5. Comparison with conventional beamforming

6. ESPRIT DOA estimation

7. Different array geometries

8. 2D DOA estimation using a planar/rhombus array

The longer-term goal is to move from this basic 1D ULA implementation toward more realistic radar and RF signal-processing simulations.

---

# 19. Why I Built This Project

I am interested in RF systems, antenna arrays, radar signal processing and sensing.

I wanted to understand DOA estimation from the basics instead of only using a ready-made implementation.

Because of that, I built this project step by step:

```text

ULA
 |
 v
Steering Vector
 |
 v
Signal Generation
 |
 v
Noise
 |
 v
Covariance Matrix
 |
 v
Eigenvalue Decomposition
 |
 v
MUSIC
 |
 v
Single Source
 |
 v
Multiple Sources

```

I will keep extending the project as I learn more about array processing and radar signal processing.
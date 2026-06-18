# An empirically-calibrated forward model for satellite-streak imagery

*Methods write-up for the sdasim empirical rendering mode. Intended as a drop-in
methods section should the pretraining experiment succeed.*

## 1. Motivation

Convolutional streak detectors pretrained on purely parametric (Gaussian-PSF,
analytic-noise) synthetic imagery transfer poorly to real ground-based optical
collects: in our experiments such pretraining gave no benefit over an ImageNet
initialization. We attribute this to a *texture* mismatch — the early layers of a
CNN respond to local second-order image statistics, and a clean Gaussian streak on
white noise has none of the structured, time-correlated texture that the atmosphere
and mount imprint on a real streak. We therefore replace the parametric forward
model with one *calibrated from real paired collects*, matching the point-spread
function, the detector noise, and — most importantly — the along- and cross-track
texture of real streaks, to the second-order statistics a detector is sensitive to.

## 2. Calibration data

Calibration uses paired collects acquired on the same instrument under matched
conditions: a **sidereal** frame (telescope tracking the stars, giving point-like
stellar PSFs) and one or more **rate** frames (tracking a target, so the stars
render as streaks of a measured length and orientation). The sidereal frame
constrains the PSF and noise; the rate frames constrain the streak texture. Detectors
are open-band silicon (≈400–950 nm), a point we return to in §5.

## 3. Point-spread function

From each sidereal frame we extract well-isolated, high-S/N, unsaturated stars,
reject cosmic rays and hot pixels with a neighbour-support test, recentre each stamp
to a common sub-pixel origin (iterated windowed centroid followed by a cubic-spline
shift), and flux-normalize. The **mean empirical PSF** $\bar K$ is the average of the
recentred stamps; it retains the real diffraction wings that a Gaussian omits. A
basis of eigen-PSFs is obtained by PCA on the residuals after projecting out the two
translation modes $\partial\bar K/\partial x,\,\partial\bar K/\partial y$ (so that
sub-pixel centroiding error does not leak into the basis).

For generation, each rendered frame draws a PSF $K = \mathrm{warp}(\bar K;\,a,e_1,e_2)$,
an affine size/ellipticity warp whose parameters $(a,e_1,e_2)$ are sampled from the
measured joint distribution (mean $\mu$, covariance $\Sigma$). Inflating the
covariance by a factor $\eta^2$ ("extend factor") generates diversity slightly beyond
the observed range, which is desirable for pretraining.

## 4. Detector noise

The noise model, fit on source-free regions, comprises: a conversion gain $g$
(e⁻/ADU, from a photon-transfer curve); a white Gaussian floor whose standard
deviation $\sigma_0$ we found to vary frame-to-frame with sky brightness, so it is
sampled per frame, $\sigma_0\!\sim\!\mathcal N(\bar\sigma_0,\,\delta\sigma_0)$; shot
noise $\sqrt{\max(S,0)/g}$ added in quadrature; a low-order background gradient; and a
sparse hot-pixel population. A two-dimensional power-spectral-density check on the
source-free regions confirmed the noise is white (no correlated-noise generator is
required). All rendering is performed at the reduced ("processed") pipeline stage,
i.e. after row/column median subtraction, so simulated and real frames are compared
at the same stage.

## 5. Streak formation and texture model

A streak is the time integral of the instantaneous PSF along the target's apparent
trajectory during the exposure $T$. We discretize it into $M\!\propto\!L$
sub-sources ($L$ = streak length in px):

$$ S(\mathbf x)\;=\;\sum_{i=1}^{M} I_i\,K\!\big(\mathbf x-\mathbf p_i\big), \qquad
\mathbf p_i=\mathbf p_0+\mathbf v\,t_i+\boldsymbol\delta(t_i). $$

A bare evaluation of this sum (constant $I_i$, fixed $K$) reproduces a Gaussian-like
streak and is what fails to transfer. The realism comes from four time-correlated
perturbations, each motivated by a distinct physical effect. Crucially, because the
detectors are **open-band** (≈1.5 octaves of wavelength), atmospheric *speckle* is
chromatically averaged out over the band; the residual atmospheric signature is
therefore **low-order and broadband** (seeing, scintillation, tip/tilt) rather than
fine high-frequency speckle. Accordingly every perturbation is driven by a
**broadband process** $\xi(t)$ with a power-law temporal spectrum
$P(f)\propto f^{-\beta}$ ($\beta\!\approx\!2$), so structure appears at all scales
rather than at a single correlation length:

1. **Scintillation** — along-track intensity flicker:
   $I_i=\bar I\,[1+\sigma_{\rm sc}\,\xi_{\rm sc}(t_i)]_+$.
2. **Tip/tilt** — a 2-D mount/atmosphere position wander
   $\boldsymbol\delta(t)=\sigma_{\rm tt}\,[\xi_\parallel(t)\,\hat{\mathbf e}_\parallel+\xi_\perp(t)\,\hat{\mathbf e}_\perp]$.
   The along-track component modulates the local path *density* (where the path
   bunches, flux piles into a smooth PSF-shaped knot), and the cross-track component
   produces the gentle centroid undulation seen as a double-humped cross-track
   structural fluctuation in real stacked profiles.
3. **Seeing-breathing** — an isotropic PSF size pulse, added as the *symmetric*
   size-derivative of the PSF so it never introduces a one-sided edge artifact:
   $S\!\mathrel{+}=\!\sum_i I_i\,\sigma_{\rm se}\,\xi_{\rm se}(t_i)\,\partial K/\partial a$.
4. **High-frequency plateau** — a post-convolution multiplicative texture field,
   $S(\mathbf x)\leftarrow S(\mathbf x)\,[1+T(\mathbf x)]$, with $T$ a band-limited
   ($k\in[k_{\rm lo},k_{\rm hi}]$ cyc/px) red-spectrum ($\propto k^{-\gamma}$) random
   field. Applied *after* convolution, this fills the resolved-but-fine texture band
   the PSF would otherwise erase; band-limiting and the red slope keep it coherent
   ("whispy") rather than white salt-and-pepper.

Because the perturbations 1–4 can drive the signal arbitrarily low and create
unphysical dark gaps in the streak core, we impose a **flux floor**: the textured
streak is clipped from below at a fraction $\phi$ of the *un-textured* streak
$S_0(\mathbf x)=\sum_i \bar I\,K(\mathbf x-\mathbf p_i)$,

$$ S(\mathbf x)\;\leftarrow\;\max\!\big(S(\mathbf x),\,\phi\,S_0(\mathbf x)\big). $$

Referencing the un-textured streak (rather than a self-dipping smoothed envelope) is
essential — it guarantees the texture *dims* but never *guts* the core, matching the
observation that real streaks always retain some flux along their length.

For moving sources the deposited PSF is the **instantaneous** (deconvolved) PSF, so
that the tip/tilt wander re-broadens the cross-section back to the measured sidereal
PSF without double counting (the deconvolution width is capped to avoid ringing).

## 6. Calibration of the texture amplitudes

We characterize real streak texture by its two-dimensional power spectral density.
Each bright real streak is rotated to a common (along-track, cross-track) frame,
high-pass filtered to isolate the texture from the smooth streak envelope,
flux-normalized, and Fourier transformed; the PSDs are averaged and compared against
a source-free noise PSD measured identically. Three features drive the model:

- the texture sits well above the noise floor out to the resolved fine-scale limit
  (≈3 px), confirming it is *signal*, not photon noise;
- the radial spectrum is **steep** (close to $k^{-4}$, far redder than the $k^{-2}$ of
  Brownian noise), consistent with low-order atmosphere and the chromatic speckle
  washout of §5;
- above the PSF cutoff ($\approx\!1/\mathrm{FWHM}$) the real PSD holds a flat
  **plateau** that a pre-convolution texture model cannot reproduce — this is the band
  the post-convolution high-frequency term (perturbation 4) is calibrated to fill.

The component amplitudes $(\sigma_{\rm sc},\sigma_{\rm tt},\sigma_{\rm se},\sigma_{\rm hf})$,
the band $[k_{\rm lo},k_{\rm hi}]$, the slopes $(\beta,\gamma)$ and the floor $\phi$
are then set so the simulated streak PSD overlays the measured one (an objective,
flux-independent criterion) while a per-knob visual sweep guards against the two
failure modes we encountered: a white (flat-spectrum) high-frequency term reads as
salt-and-pepper, and an over-amplified high-frequency or seeing term reads as a
nodular "cirrhotic" core. Representative calibrated values for our data:
$\beta\!=\!2$, $\sigma_{\rm sc}\!\approx\!0.11$, $\sigma_{\rm tt}\!\approx\!0.8$ px,
$\sigma_{\rm se}\!\approx\!0.10$, $\sigma_{\rm hf}\!\approx\!0.13$,
$[k_{\rm lo},k_{\rm hi}]\!=\![0.08,0.30]$ cyc/px, $\gamma\!=\!1.2$, $\phi\!=\!0.45$.

## 7. Generative sampling for pretraining

To generate a pretraining set, scene geometry is sampled per frame: streak **rate**
(and hence length $L=\mathrm{rate}\times T$) from the measured rate distribution
widened by the extend factor $\eta$; PSF shape from the warp distribution (§3); and
per-streak texture amplitudes jittered for diversity. Streak **orientation is sampled
uniformly on $[0,180)°$** — the full rotation, since a streak at $\theta$ and
$\theta+180°$ are identical — rather than from the measured angle distribution, which
reflects only the observing campaign's geometry and would needlessly bias the
detector. Frames are rendered in batches on the GPU with a per-scene random generator
for reproducibility.

## 8. Validation and limitations

The model is validated objectively by overlaying the simulated and real texture PSDs
(§6) and subjectively by side-by-side real/simulated/Gaussian streak galleries.
After calibration the obvious artifacts of intermediate models — disconnected beads,
salt-and-pepper grain, one-sided edge structure, nodular cores, dark holes — are
absent, and the simulated streaks reproduce the continuous, whispy, flux-floored
texture of the real data. The residual gap is subtle: a trained observer can still
distinguish real from simulated streaks. Whether this residual matters is an
empirical question, deferred to the downstream A/B transfer test (pretraining a
detector on empirical vs. Gaussian vs. ImageNet-initialized data and comparing
streak-detection metrics), which is the model's true acceptance criterion.

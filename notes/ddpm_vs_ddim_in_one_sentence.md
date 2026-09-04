# DDPM vs DDIM in one sentence

During denoising, both DDPM and DDIM strive for the marginal distribution of each visited xⱼ to match the one seen during training (p(xⱼ) ≈ q(xⱼ)), but DDPM achieves this by ensuring p(xⱼ|xⱼ₊₁) ≈ q(xⱼ|xⱼ₊₁) at every rung along the way, while DDIM realizes that p(xⱼ) ≈ q(xⱼ) can hold without the harder per-rung conditional matching being true, and so finds an update rule that achieves the former without the latter.

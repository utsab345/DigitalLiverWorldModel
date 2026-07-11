"""
Synthetic generator for the digital-liver 8-D clinical state.

Produces trajectories matching the assignment spec:
  F (fibrosis)        [0,1] ratchet, non-decreasing
  D (ductopenia)      [0,1] ratchet, irreversible
  S (strictures)      [0,1] ratchet, may step down at ERCP
  P (portal HTN)      [0,1] ratchet, non-decreasing
  A (inflammation)    [0,1] fast, mean-reverting
  C (cholestasis)     [0,1] fast, with flares
  M (malignancy)      [0,2] monotone non-decreasing
  flare               [0,1] transient, decays

Context: disease_class, age, sex, responder, UDCA start month, ERCP months.
"""

import torch

FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
F, D, S, P, A, C, M, FL = range(8)


class SyntheticGenerator:
    def __init__(self, seed=42):
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def sample_params(self, n=1):
        sus = torch.rand(n, generator=self.rng) * 0.8 + 0.1
        b_a = torch.rand(n, generator=self.rng) * 0.3 + 0.05
        b_c = torch.rand(n, generator=self.rng) * 0.3 + 0.05
        age = torch.randint(25, 70, (n,), generator=self.rng).float()
        sex = (torch.rand(n, generator=self.rng) > 0.5).float()
        dc = torch.randint(0, 4, (n,), generator=self.rng).float()
        return {
            "susceptibility": sus,
            "baseline_a": b_a,
            "baseline_c": b_c,
            "responder": (torch.rand(n, generator=self.rng) > 0.5).float(),
            "age": age,
            "sex": sex,
            "disease_class": dc,
        }

    def generate(self, n_steps=120, params=None, udca_start=12, ercp_months=None):
        if params is None:
            params = {k: v.squeeze(0) for k, v in self.sample_params(1).items()}
        if ercp_months is None:
            ercp_months = []
        sus = params["susceptibility"]
        b_a = params["baseline_a"]
        b_c = params["baseline_c"]
        responder = params["responder"]

        dc = params["disease_class"]
        dc_mult = torch.tensor([0.8, 1.0, 1.2, 1.4])[dc.long()]
        sus = sus * dc_mult

        age = params["age"]
        sex = params["sex"]
        age_factor = 1.0 + 0.002 * (age - 50)
        sex_factor = 1.0 - 0.03 * sex

        x = torch.zeros(n_steps, 8)
        x[0, F] = 0.02 + 0.05 * sus
        x[0, D] = 0.01 + 0.03 * sus
        x[0, S] = 0.05 * sus
        x[0, P] = 0.02 * sus
        x[0, A] = b_a
        x[0, C] = b_c
        x[0, M] = 0.001
        x[0, FL] = 0.0

        hazard = float(x[0, F] * x[0, C])

        for t in range(1, n_steps):
            flare = x[t - 1, FL]
            on_tx = (responder > 0.5).item() and (t >= udca_start)

            a_noise = torch.randn(1, generator=self.rng).item() * 0.03
            c_noise = torch.randn(1, generator=self.rng).item() * 0.04

            if flare > 0.1:
                a_drive = flare * 0.3 + a_noise
                c_drive = flare * 0.4 + c_noise
            else:
                a_drive = a_noise * (0.5 if on_tx else 1.0)
                c_drive = c_noise * (0.5 if on_tx else 1.0)

            a_raw = x[t - 1, A] + a_drive - 0.3 * (x[t - 1, A] - b_a)
            c_raw = x[t - 1, C] + c_drive - 0.2 * (x[t - 1, C] - b_c * (0.6 if on_tx else 1.0))
            x[t, A] = torch.clamp(a_raw, 0.0, 1.0)
            x[t, C] = torch.clamp(c_raw, 0.0, 1.0)

            drive = 0.004 * (sus + x[t, A] * x[t, C]) * (0.6 if on_tx else 1.0)
            drive = drive * age_factor * sex_factor

            x[t, F] = torch.maximum(x[t - 1, F], torch.clamp(x[t - 1, F] + drive * (1 - x[t - 1, F]), 0.0, 1.0))
            x[t, D] = torch.maximum(x[t - 1, D], torch.clamp(x[t - 1, D] + drive * 0.3 * (1 - x[t - 1, D]), 0.0, 1.0))
            x[t, P] = torch.maximum(x[t - 1, P], torch.clamp(x[t - 1, P] + drive * 0.5 * (1 - x[t - 1, P]), 0.0, 1.0))

            s_raw = x[t - 1, S]
            if t in ercp_months:
                s_raw = s_raw * 0.4
            else:
                s_raw = s_raw + drive * 0.3
            x[t, S] = torch.clamp(s_raw, 0.0, 1.0)
            if t not in ercp_months:
                x[t, S] = torch.maximum(x[t - 1, S], x[t, S])

            hazard = 0.95 * hazard + 0.05 * x[t, F] * x[t, C]
            m_drive = 0.008 * hazard * (1 + 0.5 * sus)
            x[t, M] = torch.maximum(x[t - 1, M], torch.clamp(x[t - 1, M] + m_drive, 0.0, 2.0))

            if flare > 0.5:
                x[t, FL] = flare * 0.85
            elif torch.rand(1, generator=self.rng).item() < 0.03 * (1 + 2 * x[t, C]):
                x[t, FL] = 0.5 + 0.5 * torch.rand(1, generator=self.rng).item()
            else:
                x[t, FL] = flare * 0.85

        return x

    def generate_dataset(self, n=100, n_steps=120):
        trajectories, contexts, context_tensors = [], [], []
        for _ in range(n):
            params = self.sample_params(1)
            udca = int(torch.randint(6, 36, (1,), generator=self.rng).item())
            n_ercp = int(torch.randint(0, 4, (1,), generator=self.rng).item())
            ercp = sorted([
                int(torch.randint(12, n_steps - 12, (1,), generator=self.rng).item())
                for _ in range(n_ercp)
            ])
            traj = self.generate(
                n_steps=n_steps,
                params={k: v.squeeze(0) for k, v in params.items()},
                udca_start=udca,
                ercp_months=ercp,
            )
            trajectories.append(traj)
            ctx = {
                "udca_start": udca,
                "ercp_months": ercp,
                "responder": params["responder"].squeeze(0).item(),
                "age": params["age"].squeeze(0).item(),
                "sex": params["sex"].squeeze(0).item(),
                "disease_class": params["disease_class"].squeeze(0).item(),
            }
            contexts.append(ctx)
            context_tensors.append(torch.tensor([
                params["age"].squeeze(0).item(),
                params["sex"].squeeze(0).item(),
                params["responder"].squeeze(0).item(),
                params["disease_class"].squeeze(0).item(),
                float(udca),
            ]))
        return torch.stack(trajectories), contexts, torch.stack(context_tensors)

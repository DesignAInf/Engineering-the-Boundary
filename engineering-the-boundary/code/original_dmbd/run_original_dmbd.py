"""
Re-run the detection step of Simulation 4 with the ORIGINAL DMBD code of Beck & Ramstead
(https://github.com/bayesianempirimancer/pyDMBD, MIT licence), to check the NumPy
re-implementation used in the paper.

NOTE: this script could not be executed in the environment where the paper was prepared
(PyTorch was not installable there). It is provided so that the check can be run locally:

    git clone https://github.com/bayesianempirimancer/pyDMBD
    pip install torch numpy matplotlib scipy
    cp original_dmbd/run_original_dmbd.py sim3_inversion.py mini_dmbd.py paths.py pyDMBD/   (from code/)
    cd pyDMBD && python run_original_dmbd.py

Data shape convention follows the pyDMBD README: (T, batch, n_elements, obs_dim).
"""
import numpy as np, torch
from models.DynamicMarkovBlanketDiscovery import DMBD
from sim3_inversion import gen, LEAKERS

for shield in [(), (2, 3), (1, 2, 3)]:
    Y, lab = gen(shield=shield, seed=0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    data = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)   # (T, 1, n, 1)
    model = DMBD(obs_shape=data.shape[-2:], role_dims=(1, 1, 1), hidden_dims=(2, 2, 2),
                 batch_shape=(), regression_dim=-1, control_dim=0, number_of_objects=1)
    for _ in range(10):
        model.update(data, None, None, iters=2, latent_iters=1, lr=0.5)
    p = model.assignment_pr().mean(0).squeeze()          # time-averaged q(s), q(b), q(z) per element
    pred = p.argmax(-1).numpy()
    # labels are identified up to a swap of s and z; report the better of the two matchings
    acc = max((pred == lab).mean(), (np.where(pred == 0, 2, np.where(pred == 2, 0, 1)) == lab).mean())
    print("shield", shield, "label accuracy vs target partition %.2f" % acc,
          "labels of leaking elements", pred[LEAKERS], "ELBO", float(model.ELBO()))

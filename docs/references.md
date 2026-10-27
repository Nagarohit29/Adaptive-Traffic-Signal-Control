# References

External resources this project draws on. Updated as each one is used.

## Libraries

- [Gymnasium](https://gymnasium.farama.org/): environment API
- [NumPy](https://numpy.org/): simulator and Q-table
- [matplotlib](https://matplotlib.org/): plots and the demo GIF
- [PyYAML](https://pyyaml.org/): config files
- [pytest](https://pytest.org/), [ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy-lang.org/): testing and static checks
- [Streamlit](https://streamlit.io/): deployed demo

## Background

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed.
  Chapter 6 covers Q-learning and SARSA (TD control).
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4),
  279–292. The tabular Q-learning update and its convergence result.
- Rummery, G. A. & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*.
  Technical Report CUED/F-INFENG/TR 166, Cambridge University. The origin of SARSA.
- Highway Capacity Manual: the 1800 veh/h/lane saturation flow used as the departure rate.

## Platform documentation

- Hugging Face Spaces overview: hardware and plan requirements (deployment decision)
- Streamlit Community Cloud, "Manage your app": resource limits and sleep behaviour

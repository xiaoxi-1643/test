PRNGKey
jnp.ndarray = flax.struct.field(default_factory=lambda: jnp.array([...]))

For OGbench cube-quadruple
--ogbench_dataset_dir=/data/cyh/ogbench/cube-quadruple-play-100m-v0

Format On Save

# QC
### square
python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=square-mh-low_dim --sparse=False --horizon_length=5 --exp_name=square_QC

# QC-FQL
## Robomimic 
### square
python main.py --run_group=reproduce --agent.alpha=100 --env_name=square-mh-low_dim --sparse=False --horizon_length=5 --exp_name=square_QC-FQL

# QC-FQL-MPO
## Robomimic
### square
python main.py --run_group=reproduce --agent=agents/acmpo.py --agent.alpha=100 --env_name=square-mh-low_dim --sparse=False --horizon_length=5 --exp_name=square_QC-FQL-MPO_normweight-0.1
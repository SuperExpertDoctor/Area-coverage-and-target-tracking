"""Small fixed, explicitly labeled held-out closed-loop suites."""
from pathlib import Path
from env_tracking import TrackingEnv
from src.policy_tracking import NeuralPolicy
from rollouts import run_episode,write_json


def tracking_suite(config,checkpoint,output_dir,seeds):
    import torch
    torch.set_num_threads(1)
    results=[]
    for seed in seeds:
        env=TrackingEnv(config,render_mode='json')
        try:
            record,path=run_episode(env,NeuralPolicy.from_checkpoint(checkpoint),
                seed=seed,actor_name='policy',output_dir=output_dir,
                name=f'tracking-policy-{seed}',progress_every=1800)
        finally:env.close()
        summary=record['summary']
        passed=(not summary['terminated'] and summary['visible_fraction']==1.
                and summary['min_distance']>=config.tracking_min_distance
                and summary['max_distance']<=config.tracking_range-config.tracking_range_guard)
        results.append({'seed':seed,'passed':passed,'json':str(path),'summary':summary})
    report={'checkpoint':str(Path(checkpoint).resolve()),'all_passed':all(r['passed'] for r in results),
            'results':results}
    write_json(Path(output_dir)/'suite-summary.json',report)
    return report

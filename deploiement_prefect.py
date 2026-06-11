"""
deploiement_prefect.py
----------------------
Registers Prefect deployments with schedules and starts a worker.

Run this in Terminal 2 after:
  1. Terminal 1 : prefect server start
  2. Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
                  python deploiement_prefect.py

Manual triggers (Terminal 3):
  prefect deployment run 'all/ml-pipeline-all'
  prefect deployment run 'train/ml-pipeline-train'
  prefect deployment run 'evaluate/ml-pipeline-evaluate'
  prefect deployment run 'code/ml-pipeline-code'
  prefect deployment run 'install/ml-pipeline-install'
"""

from prefect import serve
from pipeline_prefect import flow_all, flow_train, flow_evaluate, flow_code, flow_install


def main():
    # ── Deployments with cron schedules ───────────────────
    # Prefect 3: pass cron= directly to to_deployment(), no import needed

    # Full pipeline: every day at 02:00 AM
    deploy_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["mlops", "full-pipeline"],
    )

    # Train flow: every day at 03:00 AM
    deploy_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        cron="0 3 * * *",
        tags=["mlops", "training"],
    )

    # Evaluate flow: every day at 04:00 AM (after train)
    deploy_evaluate = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        cron="0 4 * * *",
        tags=["mlops", "evaluation"],
    )

    # Code quality flow: every day at 01:00 AM
    deploy_code = flow_code.to_deployment(
        name="ml-pipeline-code",
        cron="0 1 * * *",
        tags=["mlops", "code-quality"],
    )

    # Install flow: no schedule (manual only)
    deploy_install = flow_install.to_deployment(
        name="ml-pipeline-install",
        tags=["mlops", "setup"],
    )

    print("Starting Prefect worker — serving all deployments …")
    print("Open http://localhost:4200 to monitor flows.\n")

    # serve() registers deployments AND starts the worker
    serve(
        deploy_all,
        deploy_train,
        deploy_evaluate,
        deploy_code,
        deploy_install,
    )


if __name__ == "__main__":
    main()
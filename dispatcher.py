import os
import sys
import argparse
import subprocess
import threading

DEFAULT_ZONE = "us-east1-c"
TPU_NAME = "qhrr-tpu-eval"
REMOTE_HOME = "/home/Aakash"
REMOTE_PROJECT = f"{REMOTE_HOME}/qhrr_project"
MAX_WALL_TIME = 3600  # 1 hour hard ceiling per cloud execution

def _gcloud_base_args(project=None):
    """Returns common gcloud args including --project if specified."""
    args = []
    if project:
        args.extend(["--project", project])
    return args

def _stream_stdout(process):
    """
    Daemon thread: reads stdout line-by-line and prints to local terminal.
    Runs until the pipe closes or the process dies.  Because daemon=True,
    this thread is killed automatically when the main thread moves on,
    so it can never block the teardown.
    """
    try:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
    except (ValueError, OSError):
        # Pipe closed after process.kill() — expected during timeout teardown
        pass

def run_local(pretrain=False, checkpoint=None, benchmark=None, eval_subset=None):
    print("--- Dispatching to Local RTX 3070 Environment ---")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    cmd = [sys.executable, "src/inference/run_ttt.py"]
    if pretrain:
        cmd.extend(["--pretrain", "pretrain_tasks.txt"])
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if eval_subset:
        cmd.extend(["--eval-subset", str(eval_subset)])
        
    subprocess.run(cmd, env=env)

def run_cloud(project=None, pretrain=False, checkpoint=None, benchmark=None,
              eval_subset=None, keep_alive=False, reuse_existing=False, zone=DEFAULT_ZONE):
    """
    Full cloud execution pipeline with:
    - On-Demand TPU provisioning (or reuse of existing instance)
    - OS-level dead man's switch when --keep-alive is active
    - SCP upload (includes pretrain_tasks.txt or benchmark tasks + checkpoint)
    - Remote dependency bootstrap
    - SSH stdout streaming via daemon thread
    - Checkpoint rescue (pretrain mode — always executes)
    - Conditional teardown (skipped when --keep-alive is active)
    """
    print("--- Dispatching to Cloud TPU On-Demand Environment ---")
    process = None
    
    try:
        # ── Phase 1: Provision hardware ────────────────────────────────
        if reuse_existing:
            print(f"\n>>> Phase 1: SKIPPED (--reuse-existing) — using TPU '{TPU_NAME}' in {zone}")
        else:
            print("\n>>> Phase 1: Provisioning TPU VM...")
            provision_cmd = (
                f'gcloud compute tpus tpu-vm create {TPU_NAME}'
                f' --zone={zone}'
                f' --accelerator-type=v5litepod-1'
                f' --version=tpu-ubuntu2204-base'
            )
            if project:
                provision_cmd += f' --project={project}'
            subprocess.run(provision_cmd, check=True, shell=True)

        # ── Phase 1.5: Arm dead man's switch ──────────────────────────
        if keep_alive and not reuse_existing:
            print("\n>>> Phase 1.5: Arming Dead Man's Switch (shutdown -h +180)...")
            dms_cmd = (
                f'gcloud compute tpus tpu-vm ssh {TPU_NAME}'
                f' --zone={zone}'
                f' --command="sudo shutdown -h +180"'
            )
            if project:
                dms_cmd += f' --project={project}'
            subprocess.run(dms_cmd, check=True, shell=True)
            print("  ✓ Dead man's switch armed: VM will auto-terminate in 180 minutes.")
        elif reuse_existing:
            print("\n>>> Phase 1.5: SKIPPED (--reuse-existing) — dead man's switch already armed")
        
        # ── Phase 2: SCP upload ───────────────────────────────────────
        print("\n>>> Phase 2: SCP Project Architecture to VM...")
        scp_files = ["src", "tests", "requirements.txt"]
        if pretrain:
            scp_files.append("pretrain_tasks.txt")
        if benchmark:
            scp_files.append(benchmark)
        if checkpoint:
            scp_files.append(checkpoint)

        # Create remote directory first (PSCP requires it to exist)
        mkdir_cmd = (
            f'gcloud compute tpus tpu-vm ssh {TPU_NAME}'
            f' --zone={zone}'
            f' --command="mkdir -p {REMOTE_PROJECT}"'
        )
        if project:
            mkdir_cmd += f' --project={project}'
        subprocess.run(mkdir_cmd, check=True, shell=True)

        # Upload each file/directory separately to avoid PSCP path issues
        for scp_file in scp_files:
            print(f"  Uploading {scp_file}...")
            scp_command = (
                f'gcloud compute tpus tpu-vm scp --recurse'
                f' {scp_file}'
                f' {TPU_NAME}:{REMOTE_PROJECT}/'
                f' --zone={zone}'
            )
            if project:
                scp_command += f' --project={project}'
            subprocess.run(scp_command, check=True, shell=True)
        
        # ── Phase 3: Remote dependency bootstrap ─────────────────────
        print("\n>>> Phase 3: Bootstrapping Remote Dependencies...")
        install_commands = (
            f"cd {REMOTE_PROJECT} && "
            "pip install -r requirements.txt && "
            "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )
        ssh_setup_cmd = (
            f'gcloud compute tpus tpu-vm ssh {TPU_NAME}'
            f' --zone={zone}'
            f' --command="{install_commands}"'
        )
        if project:
            ssh_setup_cmd += f' --project={project}'
        subprocess.run(ssh_setup_cmd, check=True, shell=True)
        
        # ── Phase 4: Execute remote inference ────────────────────────
        print("\n>>> Phase 4: Executing Remote Pipeline...")
        if pretrain:
            run_command = (
                f"cd {REMOTE_PROJECT} && "
                "PYTHONPATH=. python3 src/inference/run_ttt.py "
                "--pretrain pretrain_tasks.txt --pretrain-epochs 5"
            )
        else:
            cmd_parts = [f"cd {REMOTE_PROJECT} && PYTHONPATH=. python3 src/inference/run_ttt.py"]
            if benchmark:
                cmd_parts.append(f"--benchmark {benchmark}")
            if checkpoint:
                cmd_parts.append(f"--checkpoint {checkpoint}")
            if eval_subset:
                cmd_parts.append(f"--eval-subset {eval_subset}")
            run_command = " ".join(cmd_parts)
        
        ssh_run_cmd = (
            f'gcloud compute tpus tpu-vm ssh {TPU_NAME}'
            f' --zone={zone}'
            f' --command="{run_command}"'
        )
        if project:
            ssh_run_cmd += f' --project={project}'
        
        # Launch SSH subprocess with stdout piped
        process = subprocess.Popen(
            ssh_run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
        
        # Stream stdout in a daemon thread — cannot block the main thread
        streamer = threading.Thread(target=_stream_stdout, args=(process,), daemon=True)
        streamer.start()
        
        # Main thread: wait with a hard wall-clock ceiling
        try:
            exit_code = process.wait(timeout=MAX_WALL_TIME)
            if exit_code != 0:
                print(f"\nWARNING: Cloud execution exited with code {exit_code}.")
        except subprocess.TimeoutExpired:
            print(f"\nFATAL: SSH wall-clock timeout ({MAX_WALL_TIME}s) exceeded. Force-killing...")
            process.kill()
            process.wait()
                
    except Exception as e:
        print(f"\nFATAL: Dispatcher pipeline crashed: {e}")
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        
    finally:
        # ── CHECKPOINT RESCUE (unconditional — always attempt) ────────
        if pretrain:
            print("\n>>> Phase 4.5: CHECKPOINT RESCUE")
            print("  Attempting to SCP core_checkpoint.pkl to local machine...")
            rescue_cmd = (
                f'gcloud compute tpus tpu-vm scp'
                f' {TPU_NAME}:{REMOTE_PROJECT}/core_checkpoint.pkl'
                f' ./core_checkpoint.pkl'
                f' --zone={zone}'
            )
            if project:
                rescue_cmd += f' --project={project}'
            try:
                subprocess.run(rescue_cmd, check=True, timeout=60, shell=True)
                size_kb = os.path.getsize("./core_checkpoint.pkl") / 1024
                print(f"  Checkpoint rescued: ./core_checkpoint.pkl ({size_kb:.1f} KB)")
            except subprocess.TimeoutExpired:
                print("  WARNING: Checkpoint rescue timed out after 60s.")
            except Exception as e:
                print(f"  WARNING: Checkpoint rescue failed: {e}")
                print("  The checkpoint may not have been written before the process ended.")

        # ── CONDITIONAL TEARDOWN ──────────────────────────────────────
        if keep_alive:
            print(f"\n>>> Phase 5: TEARDOWN SKIPPED (--keep-alive active)")
            print(f"  TPU '{TPU_NAME}' is still running in {zone}.")
            print(f"  Dead man's switch will auto-terminate in <=180 minutes.")
            print(f"  Manual teardown: gcloud compute tpus tpu-vm delete {TPU_NAME}"
                  f" --quiet --zone={zone} --project={project}")
        else:
            print("\n>>> Phase 5: TEARDOWN PROTOCOL INITIATED")
            print(f"Deleting TPU ({TPU_NAME}) -- budget protection enforced...")
            delete_command = (
                f'gcloud compute tpus tpu-vm delete {TPU_NAME}'
                f' --quiet'
                f' --zone={zone}'
            )
            if project:
                delete_command += f' --project={project}'
            try:
                subprocess.run(delete_command, check=True, timeout=120, shell=True)
                print("Teardown Successful. VM eradicated.")
            except subprocess.TimeoutExpired:
                print(f"CRITICAL WARNING: Teardown command timed out after 120s. "
                      f"You MUST delete '{TPU_NAME}' manually via GCP Console!")
            except Exception as e:
                print(f"CRITICAL WARNING: Teardown failed. "
                      f"You MUST delete '{TPU_NAME}' manually! Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="QHRRN Master Dispatcher -- routes execution to local GPU or Cloud TPU."
    )
    parser.add_argument("--local", action="store_true",
                        help="Execute on the local workstation (RTX 3070).")
    parser.add_argument("--cloud", action="store_true",
                        help="Provision an On-Demand TPU, upload code, run, and teardown.")
    parser.add_argument("--project", default=None,
                        help="GCP project ID for billing.")
    parser.add_argument("--zone", default=DEFAULT_ZONE,
                        help=f"GCP zone for TPU provisioning (default: {DEFAULT_ZONE}).")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run in pre-training mode. Saves and rescues core_checkpoint.pkl.")
    parser.add_argument("--benchmark", default=None,
                        help="Task list for benchmark evaluation.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to local checkpoint to inject.")
    parser.add_argument("--eval-subset", type=int, default=None,
                        help="Limit evaluation to the first N tasks.")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Skip TPU teardown after execution. Arms a 3-hour OS dead man's switch.")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Skip provisioning — reuse an already-running TPU instance.")
    
    args = parser.parse_args()
    
    if args.cloud:
        run_cloud(
            project=args.project,
            pretrain=args.pretrain,
            checkpoint=args.checkpoint,
            benchmark=args.benchmark,
            eval_subset=args.eval_subset,
            keep_alive=args.keep_alive,
            reuse_existing=args.reuse_existing,
            zone=args.zone
        )
    else:
        run_local(
            pretrain=args.pretrain,
            checkpoint=args.checkpoint,
            benchmark=args.benchmark,
            eval_subset=args.eval_subset
        )

if __name__ == "__main__":
    main()

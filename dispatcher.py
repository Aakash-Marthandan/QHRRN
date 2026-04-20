import os
import sys
import argparse
import subprocess
import threading

ZONE = "us-central2-b"
TPU_NAME = "qhrr-tpu"
MAX_WALL_TIME = 3600  # 1 hour hard ceiling per cloud execution

def _gcloud_base_args(project=None):
    """Returns common gcloud args including --project if specified."""
    args = ["--zone", ZONE]
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

def run_local():
    print("--- Dispatching to Local RTX 3070 Environment ---")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run([sys.executable, "src/inference/run_ttt.py"], env=env)

def run_cloud(project=None, pretrain=False):
    """
    Full cloud execution pipeline with:
    - Spot TPU provisioning
    - SCP upload (includes pretrain_tasks.txt when in pretrain mode)
    - Remote dependency bootstrap
    - SSH stdout streaming via daemon thread
    - Checkpoint rescue (pretrain mode only)
    - Unconditional teardown
    """
    print("--- Dispatching to Cloud TPU Spot Environment ---")
    base_args = _gcloud_base_args(project)
    process = None
    
    try:
        # 1. Provision hardware via Spot discounts
        print("\n>>> Phase 1: Provisioning TPU VM...")
        env = os.environ.copy()
        if project:
            env["GCP_PROJECT"] = project
        subprocess.run(["bash", "scripts/deploy_tpu.sh"], check=True, env=env)
        
        # 2. SCP upload
        print("\n>>> Phase 2: SCP Project Architecture to VM...")
        scp_files = ["src", "tests", "requirements.txt"]
        if pretrain:
            scp_files.append("pretrain_tasks.txt")

        scp_command = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            "--recurse"
        ] + scp_files + [
            f"{TPU_NAME}:~/qhrr_project/"
        ] + base_args
        subprocess.run(scp_command, check=True)
        
        # 3. Remote dependency bootstrap
        print("\n>>> Phase 3: Bootstrapping Remote Dependencies...")
        install_commands = (
            "cd ~/qhrr_project && "
            "pip install -r requirements.txt && "
            "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )
        ssh_setup_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
            "--command", install_commands
        ] + base_args
        subprocess.run(ssh_setup_cmd, check=True)
        
        # 4. Execute remote inference
        print("\n>>> Phase 4: Executing Remote Pipeline...")
        if pretrain:
            run_command = (
                "cd ~/qhrr_project && "
                "PYTHONPATH=. python3 src/inference/run_ttt.py "
                "--pretrain pretrain_tasks.txt --pretrain-epochs 5"
            )
        else:
            run_command = (
                "cd ~/qhrr_project && "
                "PYTHONPATH=. python3 src/inference/run_ttt.py"
            )
        
        ssh_run_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
            "--command", run_command
        ] + base_args
        
        # Launch SSH subprocess with stdout piped
        process = subprocess.Popen(
            ssh_run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
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
        # ── CHECKPOINT RESCUE (before teardown) ────────────────────────
        if pretrain:
            print("\n>>> Phase 4.5: CHECKPOINT RESCUE")
            print("  Attempting to SCP core_checkpoint.pkl to local machine...")
            rescue_cmd = [
                "gcloud", "compute", "tpus", "tpu-vm", "scp",
                f"{TPU_NAME}:~/qhrr_project/core_checkpoint.pkl",
                "./core_checkpoint.pkl"
            ] + base_args
            try:
                subprocess.run(rescue_cmd, check=True, timeout=60)
                size_kb = os.path.getsize("./core_checkpoint.pkl") / 1024
                print(f"  Checkpoint rescued: ./core_checkpoint.pkl ({size_kb:.1f} KB)")
            except subprocess.TimeoutExpired:
                print("  WARNING: Checkpoint rescue timed out after 60s.")
            except Exception as e:
                print(f"  WARNING: Checkpoint rescue failed: {e}")
                print("  The checkpoint may not have been written before the process ended.")

        # ── UNCONDITIONAL TEARDOWN ─────────────────────────────────────
        print("\n>>> Phase 5: TEARDOWN PROTOCOL INITIATED")
        print(f"Deleting TPU ({TPU_NAME}) -- budget protection enforced...")
        delete_command = [
            "gcloud", "compute", "tpus", "tpu-vm", "delete", TPU_NAME,
            "--quiet"
        ] + base_args
        try:
            subprocess.run(delete_command, check=True, timeout=120)
            print("Teardown Successful. Spot VM eradicated.")
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
                        help="Provision a Spot TPU, upload code, run, and teardown.")
    parser.add_argument("--project", default=None,
                        help="GCP project ID for billing.")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run in pre-training mode. Saves and rescues core_checkpoint.pkl.")
    
    args = parser.parse_args()
    
    if args.cloud:
        run_cloud(project=args.project, pretrain=args.pretrain)
    else:
        run_local()

if __name__ == "__main__":
    main()

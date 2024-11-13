#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This wraps wandb to write logs locally as well."""

import os
from typing import Any, Dict, Optional, Union
import uuid
import json
import atexit
import sys

import wandb

log_file = None
DEFAULT_PROJET = 'default_project'
self_module = sys.modules[__name__]

def sync_attributes():
    """Syncs this module's attributes and wandb attributes"""
    # Wrap every wandb attribute
    for attribute in dir(wandb):
        # We don't want to overwrite wandb_wrapper.{log,finish}
        if attribute not in ['log', 'finish']:
            setattr(self_module, attribute, getattr(wandb, attribute))

sync_attributes()

def init(
    out_path: str = 'out',
    **wandb_kwargs
) -> Union[wandb.sdk.wandb_run.Run, wandb.sdk.lib.RunDisabled]:
    """
    Initializes wandb as well as the local log file.

    @param out_path: The path where the logs will be written.
    """
    run = wandb.init(**wandb_kwargs)

    # Some attributes (for example wandb.run) are changed after init
    sync_attributes()

    # If we use wandb, share the run id and name with wandb. Otherwise generate a
    #  random run id.
    run_id = None
    # See https://community.wandb.ai/t/how-do-i-query-whether-a-run-object-is-disabled-rundisabled/2256/8
    #  on how to check whether wandb is enabled
    if not isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
        run_id = run.id
        project = run.project
        name = run.name
    else:
        run_id = str(uuid.uuid4())
        project = wandb_kwargs.get('project', DEFAULT_PROJET)
        # If a name is provided, use that instead of randomly generating one.
        name = wandb_kwargs.get('name', uuid.uuid4())

    # pylint: disable=W0603
    # yeah, yeah, global isn't great but this helps us mirror the wandb api
    global log_file
    log_dir_path = os.path.join(out_path, f'{project}/{name}/{run_id}')
    os.makedirs(log_dir_path, exist_ok=True)
    log_path = os.path.join(log_dir_path, 'log.txt')
    print(f'wandb_wrapper is logging to {log_path}')
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(f'{wandb_kwargs}\n')
    atexit.register(finish)

    return run

def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None:
    """Logs to wandb and the log file"""
    if step or step == 0:
        data['step'] = step
    log_file.write(f'{json.dumps(data)}\n')
    wandb.log(data, step, commit, sync)

def finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None:
    """Finish wandb and close the local log file."""
    print('Finishing up wandb_wrapper')
    log_file.close()
    wandb.finish(exit_code, quiet)
    atexit.unregister(finish)

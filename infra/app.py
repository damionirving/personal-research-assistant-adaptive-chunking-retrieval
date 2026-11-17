#!/usr/bin/env python3
import os

from aws_cdk import App, Environment

from pra_infra.stack import PersonalResearchAssistantStack


app = App()

PersonalResearchAssistantStack(
    app,
    "PersonalResearchAssistantStack",
    env=Environment(
        account=os.environ.get("PRA_AWS_ACCOUNT_ID") or app.node.try_get_context("account"),
        region=os.environ.get("PRA_AWS_REGION") or app.node.try_get_context("region") or "us-east-1",
    ),
)

app.synth()


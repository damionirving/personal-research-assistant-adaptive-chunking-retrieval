#!/usr/bin/env python3
from aws_cdk import App, Environment

from pra_infra.stack import PersonalResearchAssistantStack


app = App()

PersonalResearchAssistantStack(
    app,
    "PersonalResearchAssistantStack",
    env=Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)

app.synth()


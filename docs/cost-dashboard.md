# Cost Dashboard

This document describes the CloudWatch-based cost dashboard and alarm setup for the Personal Research Assistant.

## What It Is

The dashboard lives entirely inside AWS.

It uses:

- CloudWatch Dashboard for the visual panels
- CloudWatch Alarms for threshold alerts
- SNS email notifications for alarm delivery

The stack creates the dashboard in the same CDK deployment that defines the API, Lambda, Cognito, and S3 resources.

## What It Shows

The dashboard is meant to answer a few practical questions quickly:

1. How much traffic is the app seeing?
2. Are errors, latency, or retries increasing?
3. Is the app drifting into expensive behavior?

The first version focuses on native AWS signals that already exist, so it stays lightweight and maintainable.

### Current Panels

- request volume
- Lambda invocation volume
- Lambda errors
- API Gateway 4xx and 5xx errors
- Lambda and API latency
- throttles and integration latency

## Why This Is Useful

This is not just a pretty graph page. It is an operational control surface.

It helps you:

- notice spend-related behavior early
- spot retry storms and model access failures
- see whether latency is climbing
- identify whether uploads, chat, or the Lambda layer are driving load
- keep the app focused instead of letting traffic and model calls drift out of bounds

## Alarms

The stack currently wires three CloudWatch alarms to SNS email notifications:

- API Lambda error alarm
- API Lambda duration alarm
- API Gateway 5xx alarm

These alarms are useful because cost and reliability problems often show up together.

Examples:

- repeated Bedrock failures can raise both latency and errors
- an accidental traffic burst can increase request count and costs
- a misconfiguration can create a visible 5xx pattern before users report it

## Email Notifications

Alarm notifications are sent to the configured owner email address.

The CDK stack exposes this as a context value:

```text
owner_email
```

The stack also accepts:

```text
aws_owner_email
```

If neither is supplied, the stack falls back to the default configured in `infra/pra_infra/stack.py`.

## How To Use It

After deployment:

1. Open the CloudWatch dashboard link from the CDK outputs.
2. Watch request volume, latency, and error panels together.
3. Use alarms as the early warning system.
4. Check CloudWatch Logs when an alarm fires to see the underlying error detail.

For alarms:

- an email subscription must be confirmed once
- AWS will send a confirmation email to the subscribed address
- after confirmation, future alerts go to that mailbox automatically

## How To Run It

The dashboard is created by CDK when you deploy the stack.

Recommended deploy pattern:

```cmd
cd infra
cdk.cmd deploy --context owner_email=<your-alert-email> --force
```

The same stack deployment also creates the SNS topic, email subscription, alarms, and dashboard.

## Practical Guidance

Use the dashboard to watch for:

- spikes in request count
- elevated Lambda duration
- repeated server errors
- throttles
- possible cost drift caused by more traffic or more retries

Use the alarms to catch:

- outage conditions
- performance regressions
- usage anomalies
- expensive request bursts

## Limitations

This dashboard is a strong first step, but it is still an AWS operational view, not a full billing analytics system.

It does not yet provide:

- per-user spend
- monthly invoice breakdowns
- product analytics
- long-term cost forecasting

Those can come later if the app grows into a broader product.

## Recommended Next Evolution

If you want to expand it later, the next logical additions would be:

- Bedrock usage metrics
- per-route cost estimates
- stronger upload/chat utilization tracking
- AWS Budgets for monthly spend guardrails
- a simple user-facing usage page if you later add plans or a paywall

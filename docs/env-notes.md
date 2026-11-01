# Personal Research Assistant Notes on `.env.example`

## Question

**Are the values in `.env.example` used by the actual MVP application hosted on AWS?**

## Answer

No. The values in `.env.example` are primarily intended for **local development** and as a **configuration template**.

For the AWS-hosted MVP, the application uses configuration supplied by AWS during deployment. Runtime values come from:

- AWS CDK (`infra/pra_infra/stack.py`)
- Lambda environment variables configured by CDK
- Amazon Cognito
- AWS Secrets Manager
- Amazon S3
- Other AWS resources provisioned by the CDK stack

That same rule applies now that the website integration can be served from Firebase. The static files in `data/research-assistant/` are part of the repo and can be deployed to Firebase Hosting, but the protected backend behavior still comes from the AWS deployment, not from `.env.example`.

## Rule of Thumb

| Environment | Configuration Source |
| --- | --- |
| Local development | `.env.example` copied to `.env` (or equivalent local configuration) |
| AWS deployment | CDK stack configuration + AWS-managed runtime configuration |

## Important

- `.env.example` is **not** read directly by the production AWS deployment.
- `.env.example` is also **not** the production configuration source for the Firebase-hosted page; production website behavior depends on the checked-in front-end code plus the deployed AWS API, Cognito configuration, and CORS allowlist.
- It serves as documentation and a template for local development.
- Some values may be the same in both environments, but only because they were intentionally configured to match.
- To change production behavior, update the CDK infrastructure and/or the AWS configuration, not `.env.example`.

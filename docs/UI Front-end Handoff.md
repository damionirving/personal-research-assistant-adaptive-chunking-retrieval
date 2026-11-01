# UI Front-end Handoff

This document lists the safe front-end assets to hand to a UI developer who needs to recreate the MVP as a webpage inside another site while matching the current look and feel.

## What to share

- [apps/api/src/pra_api/templates/index.html](../apps/api/src/pra_api/templates/index.html)
- [apps/api/src/pra_api/static/app.css](../apps/api/src/pra_api/static/app.css)
- [apps/api/src/pra_api/static/favicon.ico](../apps/api/src/pra_api/static/favicon.ico)
- [apps/api/src/pra_api/static/favicon.svg](../apps/api/src/pra_api/static/favicon.svg)
- [apps/api/src/pra_api/static/favicon-96x96.png](../apps/api/src/pra_api/static/favicon-96x96.png)
- [apps/api/src/pra_api/static/apple-touch-icon.png](../apps/api/src/pra_api/static/apple-touch-icon.png)
- [data/research-assistant/index.html](../data/research-assistant/index.html)
- [data/research-assistant/app.js](../data/research-assistant/app.js)

The `data/research-assistant/` pair is the current Firebase-hosted Tractlytics integration. `index.html` provides the page shell and controls, while `app.js` wires sign-in, upload, chat, RAG evaluation, recent-evaluation polling, and status rendering to the deployed AWS API.

## What not to share

- [apps/api/src/pra_api/static/app.js](../apps/api/src/pra_api/static/app.js), unless it is first sanitized to remove backend wiring, Cognito auth flow, API route calls, trace IDs, and AWS-specific request handling.
- `infra/`
- `platform/core/`
- `apps/api/src/pra_api/rag_worker.py`
- `apps/api/src/pra_api/lambda_handler.py`

## What the UI developer should preserve

- The dark blue and teal visual language.
- The compact card-and-panel layout.
- The simple upload, sign-in, sign-out, and chat controls.
- The session label and status display.
- The assistant-bubble metadata treatment, including timestamps and compact debug lines.
- The feature-flagged RAG performance and RAGAS evaluation buttons, progress indicators, and hidden states.

## Styling notes

- Keep the current button sizing, spacing, and rounded corners.
- Keep the restrained accent colors and avoid switching to a generic light theme.
- Preserve the favicon and touch icon set so the page still feels like the same product.
- Match the current mobile behavior, where the layout collapses to a single column and the controls stack vertically.

## Integration note

The handoff should be treated as a visual and interaction reference only. The UI developer can connect it to their own backend endpoints or mock data sources, but the AWS deployment code, Lambda handlers, and infrastructure wiring should stay out of the handoff package.

For the current Tractlytics-hosted integration, the AWS API must allow these browser origins through CORS:

- `http://localhost:5000`
- `http://127.0.0.1:5000`
- `https://<your-domain.com>`
- `https://www.<your-domain.com>`
- `https://tractlyticswebsite.web.app`
- `https://tractlyticswebsite.firebaseapp.com`

Because Cognito redirects back to the hosted page itself, the Cognito app client must also allow these callback and logout URLs:

- `http://localhost:5000/research-assistant/`
- `http://127.0.0.1:5000/research-assistant/`
- `https://<your-domain.com>/research-assistant/`
- `https://www.<your-domain.com>/research-assistant/`
- `https://tractlyticswebsite.web.app/research-assistant/`
- `https://tractlyticswebsite.firebaseapp.com/research-assistant/`

# District · an interactive drive

An eighty-car district for exploring how a decision changes a journey. All cars depart together. This prototype has no LLM calls, learned negotiators, credits or training.

Run `./start`, then open [the demonstration](http://127.0.0.1:8771/layout?demo=1).

1. Hover a car to see its number, destination and route. Click it to start following it.
2. Cards distinguish incoming requests, your outgoing requests and voluntary offers. Each shows one action and side-by-side time effects for you and the involved cars. Available exchanges and incoming requests appear as driving continues. Hover a choice to see translucent future cars, a red interaction marker and their projected paths. The speculative future runs at 8× while the actual drive keeps its selected speed.
3. Click a choice to take it. The two real drivers receive the passage instruction; the changed world continues afterwards. There is no clip, automatic return or automatic speed change.

Pause, change driving speed, drag the map, zoom, or return to following the car. Click another car to follow it. **New drive** resets the whole demonstration. Keyboard focus exposes the same car information and future previews; Enter chooses a car/action. Cards keep their positions while being inspected; expired choices become unavailable rather than turning into a different action under the pointer.

Opportunities follow actual or predicted delay causes, including cars already stopped at a yield line when the other driver can still safely hold. Choices can exchange passage priority, reserve an earlier gap, or ask a priority holder to release the head of your queue. They preserve the queue's lane order. Other cars initiate heuristic requests, which you can accept or decline; they do not accept agreements autonomously.

## What the numbers mean

The effects are approximate **local delay changes**, relative to ordinary driving. They are conditional on current positions, announced paths and ordinary driving by other cars. A displayed benefit does not guarantee a crossing-time or destination-time saving.

Crossing times within the 25-second forecast are interpolated. When a crossing is outside that window, the estimator compares progress at a common horizon and converts the difference into a rough time equivalent using road speed. Cards round the effects for quick comparison. Exact values, rough confidence and forecast scope remain in the optional inspection details. Later traffic, negotiations and decisions can interrupt either forecast. Successful passage is observed separately from its predicted effect.

The continuing drive retains the existing bicycle engine, 0.1-second controls, 0.02-second physics, continuous collision checks and road containment. Future previews use the existing faster lane-motion approximation. Both use the conventional driving policy, with an additional binding passage constraint when an agreement is made. The help panel explains this distinction.

Independent passage agreements can run together when their participants and reserved road space are separate and no known circular wait is created. Existing promises remain binding when changing the followed car or accepting another deal; both ordinary and alternative forecasts include them. Once delivered, later decisions operate on the changed world. A physical fault stops and preserves the drive; it cannot be presented as delivery. Straight-road overtaking remains unsupported by the current single-lane-per-direction roads.

A passage swap cannot sell an overtake on a shared approach. Feasibility checks follow the announced corridor through the holding area, including bends, even when the buyer is still reported as driving. Forecast checks retain indirect queues and binding hold dependencies when another car masks the displayed reason. Detecting this lane-order impossibility during execution stops and records a fault; the promise is neither silently cancelled nor counted as delivered.

## Implementation

- `demos/network/layout.html` and `interactive.js`: one map-first application, car picking, follow camera, hover ghosts and persistent decisions. Forecast computation never owns the viewer clock.
- `src/vehicle_days/interactive.py`: evolving physical world, observed-state snapshots, versioned decisions, incoming requests, concurrent agreements and local estimates. `POST /api/demo` supports `new`, `advance`, `options`, `future`, `apply` and `decline`.
- `driving.py`, `driving_roads.py`, `layout_traffic.py`: retained physics and conventional controls. Exact-pose and conservatively certified pavement caches speed repeated body checks; failed certificates use the original road-union calculation. A proven distance bound skips irrelevant parked-car SAT checks.
- `reasoning.py`, `encounters.py`, `forecast_stream.py`: shared public-state forecasts, causal queue reasoning and retained research inspection interfaces. Forecast processes are shared, capped at sixteen; live mode allows ten outstanding jobs with bounded caches. Lookahead spacing adapts to measured job cost and play speed. Forecasts refresh asynchronously; the optional forecast details retain observation times and estimate age, while current feasibility is checked again before application.

Only the initial frame, road geometry and route definitions are imported from the ordinary reference. Later positions and arrivals are generated by the physical simulation. Hovering does not alter it. Applying revalidates the current passage, increments the world revision, discards unobserved buffered continuation and generates new physical motion. Old-revision responses cannot become current actions or ghosts. Delivery/arrival events are displayed only when the player's clock reaches them.

Actual demonstration drives are saved on reset/close, and faults immediately, under `data/community/demonstrations/`. Each recording distinguishes the observed interval from extra buffered continuation. Unused preparation does not create a drive record.

The original reference and prior physical comparison records remain under `data/community/`. The replaced interface and its documentation were preserved under `data/community/verification/before-interactive/`. Historical research/assets were preserved in `/Users/ahmedhamedi/Dev/Imperial/Thesis-history/` before the earlier product consolidation.

## Verification

Run `PYTHONPATH=src .venv/bin/python -m pytest -q`.

Checks cover safe stopped-car choices, upstream queue causes, incoming requests and declines, concurrent agreements and exactly-once delivery, preservation of existing promises in forecasts, real continuation changes, read-only futures, rejected stale actions, no reference-future inputs, finite labelled estimates, bounded background work and the existing engine/contract checks. Browser verification records and the latest test output live under `data/community/verification/`.

This is an inspectable interaction prototype. It is not evidence of learned negotiation, emergent cooperation or improved traffic efficiency. No new training or traffic-efficiency experiment is part of this change.

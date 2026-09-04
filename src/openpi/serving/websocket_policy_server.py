import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Implements `infer` (optionally with a caller-supplied flow-matching `noise`,
    for DSRL latent-space RL) and `get_prefix_rep`.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                message = msgpack_numpy.unpackb(await websocket.recv())

                # DSRL envelope (upstream dsrl_pi0's wire format), backward-compatible
                # by construction: a plain observation dict carries no "method"/"obs"
                # keys, so existing clients (avantbot's FrankaEEPi05Agent, the SubRL
                # loop, openpi's own examples) fall through to `infer` on the raw dict
                # exactly as before. Only a client that WANTS noise or the prefix
                # readout wraps its payload.
                method = message.get("method", "infer") if isinstance(message, dict) else "infer"
                obs = message.get("obs", message) if isinstance(message, dict) else message

                infer_time = time.monotonic()
                if method == "infer":
                    # Noise rides INSIDE obs (upstream convention) so it survives any
                    # client that only knows how to build an observation dict. Pop it
                    # before the input transforms ever see it.
                    noise = obs.pop("noise", None) if isinstance(obs, dict) else None
                    # Keyword-only on our Policy.infer; a positional call is a TypeError.
                    action = self._policy.infer(obs) if noise is None else self._policy.infer(obs, noise=noise)
                elif method == "get_prefix_rep":
                    action = self._policy.get_prefix_rep(obs)
                elif method == "reset":
                    self._policy.reset()
                    action = {}
                else:
                    raise ValueError(f"Unknown method {method!r}; expected infer/get_prefix_rep/reset.")
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

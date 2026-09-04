import logging
import time
from typing import Any, Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict, *, noise: Optional[Any] = None) -> Dict:  # noqa: UP006
        """Run inference. `noise` (DSRL) seeds the flow-matching integration.

        Shape is `(action_horizon, action_dim)` or `(1, action_horizon, action_dim)` —
        for pi0.5 that is `(50, 32)`. A fixed noise makes the returned action chunk a
        deterministic function of the observation; `noise=None` keeps the server's own
        RNG draw AND the plain-dict wire format, so nothing about the existing
        rollout/eval path changes.
        """
        if noise is None:
            # Byte-identical to the pre-DSRL client: works against old servers too.
            data = self._packer.pack(obs)
        else:
            data = self._packer.pack({"method": "infer", "obs": {**obs, "noise": noise}})
        return self._request(data)

    def get_prefix_rep(self, obs: Dict) -> Any:  # noqa: UP006
        """DSRL z_rl: the last prefix slot's embedding, float32 `[1, emb]`.

        Upstream dsrl_pi0 pulls the full `[b, s, emb]` prefix tensor and slices
        `[:, -1, :]` itself; this server does the slice before packing, so the caller
        gets the finished feature. Cheaper than `infer` — no denoise loop.
        """
        response = self._request(self._packer.pack({"method": "get_prefix_rep", "obs": obs}))
        return response["prefix_rep"]

    def _request(self, data: bytes) -> Dict:  # noqa: UP006
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

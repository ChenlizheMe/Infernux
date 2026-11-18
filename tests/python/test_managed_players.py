"""The Host's owned process uses the existing ready/request/response protocol."""
import sys
from infernux.host.context import configure
from infernux.host.players import ManagedPlayers


def test_ready_command_response_shutdown_and_relaunch(tmp_path):
    configure(tmp_path)
    child = tmp_path / "controlled.py"
    child.write_text('''
import json, os, time
from pathlib import Path
ready = Path(os.environ['_INFERNUX_READY_FILE'])
request = Path(os.environ['_INFERNUX_PLAYER_CONTROL_FILE'])
response = Path(os.environ['_INFERNUX_PLAYER_RESPONSE_FILE'])
token = os.environ['_INFERNUX_PLAYER_CONTROL_TOKEN']
ready.write_text('ENGINE_LOADED', encoding='utf-8')
seen = ''
while True:
    if not request.exists():
        time.sleep(0.002)
        continue
    value = json.loads(request.read_text(encoding='utf-8'))
    if value['command_id'] == seen:
        time.sleep(0.002)
        continue
    seen = value['command_id']
    assert value['token'] == token
    temporary = response.with_suffix('.tmp')
    temporary.write_text(json.dumps({'command_id':seen, 'ok':True, 'data':{'action':value['action']}}), encoding='utf-8')
    temporary.replace(response)
    if value['action'] == 'shutdown': break
''', encoding="utf-8")
    player = ManagedPlayers()
    try:
        first = player.launch(sys.executable, (str(child),), timeout=5)
        assert first["running"] and first["ready"]
        assert player.control("observe", timeout=5) == {"action": "observe"}
        assert not player.stop(timeout=5)["running"]
        second = player.launch(sys.executable, (str(child),), timeout=5)
        assert second["pid"] != first["pid"] and second["ready"]
        assert not player.stop(timeout=5)["ready"]
    finally:
        if player.process is not None and player.process.poll() is None:
            player.process.terminate()
            player.process.wait()
        player.close_service()

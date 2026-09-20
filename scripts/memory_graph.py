"""Project session_memory.json into a live, local relationship graph."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "memory_graph.html"


def _items(value, name):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Session memory field '{}' must be a list".format(name))
    return value


def project(memory):
    """Return a browser-friendly graph without changing memory semantics."""
    if not isinstance(memory, dict):
        raise ValueError("Session memory must be a JSON object")
    turns = _items(memory.get("turns"), "turns")
    working = _items(memory.get("working_memory"), "working_memory")
    if any(not isinstance(item, dict) for item in turns):
        raise ValueError("Every completed turn must be an object")
    if any(not isinstance(item, dict) for item in working):
        raise ValueError("Every working-memory item must be an object")
    scene = memory.get("scene") or {}
    if not isinstance(scene, dict):
        raise ValueError("Session memory field 'scene' must be an object")
    source = memory.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("Session memory field 'source' must be an object")

    speakers = []
    for name in [*_items(memory.get("participants"), "participants"),
                 *(item.get("speaker") for item in turns),
                 *(item.get("speaker") for item in working)]:
        if isinstance(name, str) and name and name not in speakers:
            speakers.append(name)

    topic_info = {}
    for item in _items(scene.get("topics"), "scene.topics"):
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        name = str(item["name"]).strip()
        topic_info.setdefault(name, dict(summary="", evidence_turn_ids=[]))
        topic_info[name]["summary"] = str(item.get("summary", "")).strip()
        topic_info[name]["evidence_turn_ids"] = _items(
            item.get("evidence_turn_ids"), "scene.topics.evidence_turn_ids")
    for turn in turns:
        for name in _items(turn.get("topic_names"), "turn.topic_names"):
            if isinstance(name, str) and name.strip():
                topic_info.setdefault(name.strip(), dict(summary="", evidence_turn_ids=[]))
    if turns and not topic_info:
        topic_info["未归类话题"] = dict(summary="尚未识别出明确话题", evidence_turn_ids=[])
    evidence_topics = {}
    for name, info in topic_info.items():
        for turn_id in info["evidence_turn_ids"]:
            evidence_topics.setdefault(turn_id, []).append(name)

    nodes = []
    edges = []
    node_ids = set()

    def add_node(identifier, kind, label, details, turn_ids=None):
        if identifier in node_ids:
            return
        node_ids.add(identifier)
        nodes.append(dict(id=identifier, type=kind, label=label,
                          details=details, turn_ids=turn_ids or []))

    for speaker in speakers:
        completed = [item.get("id") for item in turns if item.get("speaker") == speaker]
        pending = [item for item in working if item.get("speaker") == speaker]
        add_node("speaker:" + speaker, "speaker", speaker, {
            "已完成话轮": len(completed), "待完成发言": len(pending),
        }, completed)

    for index, (name, info) in enumerate(topic_info.items(), 1):
        add_node("topic:{:04d}".format(index), "topic", name, {
            "话题摘要": info["summary"] or "暂无摘要",
            "证据话轮": info["evidence_turn_ids"],
        }, list(info["evidence_turn_ids"]))
    topic_ids = {node["label"]: node["id"] for node in nodes if node["type"] == "topic"}

    intent_nodes = {}
    for turn in turns:
        if not isinstance(turn, dict) or not str(turn.get("id", "")).strip():
            raise ValueError("Every completed turn requires a non-empty id")
        turn_id = str(turn["id"])
        speaker = str(turn.get("speaker") or "未识别人物")
        summary = str(turn.get("summary") or "暂无观点摘要").strip()
        opinion_id = "opinion:" + turn_id
        start = turn.get("start_s")
        end = turn.get("end_s")
        time_range = "{:.2f}–{:.2f}s".format(float(start), float(end)) if start is not None and end is not None else "未知"
        add_node(opinion_id, "opinion", summary, {
            "人物": speaker, "话轮": turn_id, "时间": time_range,
            "言语行为": turn.get("speech_act") or "other",
            "立场": turn.get("stance") or "unclear",
            "原始贡献": _items(turn.get("contribution_ids"), "turn.contribution_ids"),
        }, [turn_id])
        if "speaker:" + speaker not in node_ids:
            add_node("speaker:" + speaker, "speaker", speaker,
                     {"已完成话轮": 1, "待完成发言": 0}, [turn_id])
        edges.append(dict(source="speaker:" + speaker, target=opinion_id,
                          type="EXPRESSES", label="表达", turn_id=turn_id))

        names = [name.strip() for name in _items(turn.get("topic_names"), "turn.topic_names")
                 if isinstance(name, str) and name.strip()]
        names = names or evidence_topics.get(turn_id, [])
        if not names:
            names = ["未归类话题"]
            if names[0] not in topic_ids:
                identifier = "topic:{:04d}".format(len(topic_ids) + 1)
                add_node(identifier, "topic", names[0], {"话题摘要": "尚未识别出明确话题", "证据话轮": []})
                topic_ids[names[0]] = identifier
        for name in names:
            edges.append(dict(source=topic_ids[name], target=opinion_id,
                              type="ABOUT", label="关于", turn_id=turn_id))

        intent = str(turn.get("intent") or "意图待识别").strip()
        if intent not in intent_nodes:
            intent_id = "intent:{:04d}".format(len(intent_nodes) + 1)
            intent_nodes[intent] = intent_id
            add_node(intent_id, "intent", intent, {"相关人物": [], "证据话轮": []})
        intent_id = intent_nodes[intent]
        intent_node = next(node for node in nodes if node["id"] == intent_id)
        if speaker not in intent_node["details"]["相关人物"]:
            intent_node["details"]["相关人物"].append(speaker)
        intent_node["details"]["证据话轮"].append(turn_id)
        intent_node["turn_ids"].append(turn_id)
        edges.append(dict(source=opinion_id, target=intent_id,
                          type="INTENDS", label="意图", turn_id=turn_id))

    pending = [dict(speaker=item.get("speaker") or "未识别人物",
                    text=item.get("text") or "",
                    start_s=item.get("start_s"), end_s=item.get("end_s"))
               for item in working if isinstance(item, dict)]
    return {
        "session": {
            "id": memory.get("session_id") or "未命名会话",
            "created_at": memory.get("created_at"),
            "summary": scene.get("session_summary") or "暂无会话摘要",
            "source": source.get("dialogue_state"),
        },
        "counts": {
            "speakers": len([node for node in nodes if node["type"] == "speaker"]),
            "topics": len([node for node in nodes if node["type"] == "topic"]),
            "opinions": len(turns), "intents": len(intent_nodes),
            "pending": len(pending),
        },
        "nodes": nodes, "edges": edges, "working_memory": pending,
    }


def load_graph(path):
    return project(json.loads(path.read_text(encoding="utf-8")))


def serve(path, host="127.0.0.1", port=8765, open_browser=True):
    source = Path(path).expanduser().resolve(strict=True)
    if not PAGE.is_file():
        raise OSError("Graph page is missing: {}".format(PAGE))
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, body=b"", etag=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            if etag:
                self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route in {"/", "/index.html"}:
                return self._send(200, "text/html; charset=utf-8", page)
            if route == "/favicon.ico":
                return self._send(204, "image/x-icon")
            if route != "/api/graph":
                return self._send(404, "text/plain; charset=utf-8", "Not found\n".encode())
            try:
                stat = source.stat()
                etag = '"{}-{}"'.format(stat.st_mtime_ns, stat.st_size)
                if self.headers.get("If-None-Match") == etag:
                    return self._send(304, "application/json", etag=etag)
                body = json.dumps(load_graph(source), ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body, etag)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(422, "application/json; charset=utf-8", body)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    url = "http://{}:{}".format(host, server.server_address[1])
    print("Watching {}".format(source), flush=True)
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def run(args):
    return serve(args.session_memory, args.host, args.port, not args.no_open)

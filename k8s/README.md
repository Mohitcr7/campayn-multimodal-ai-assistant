# Kubernetes deployment

Five workloads in one namespace: **Postgres** (durable state), **Redis** (history + cache), **API** (2 replicas), **UI**.

```
                    ┌──────────────┐
  browser ─NodePort─▶  campayn-ui  │  Streamlit :8501
                    └──────┬───────┘
                           │ http://campayn-api:8000
                    ┌──────▼───────┐
                    │ campayn-api  │  FastAPI :8000  (replicas: 2)
                    └──┬────────┬──┘
                       │        │
              ┌────────▼──┐  ┌──▼─────┐
              │ postgres  │  │ redis  │
              │ (PVC 1Gi) │  │ (LRU)  │
              └───────────┘  └────────┘
```

**What lives where**

| Store | Holds | Durable? |
|---|---|---|
| Postgres | LangGraph checkpoints, feedback, request log | yes — PVC |
| Redis | conversation history (1h), intent cache (24h) | no — disposable by design |
| Image | 56k creator profiles (CSV), ChromaDB vector store | baked at build time |

## Cost

There is **no free managed Kubernetes**. EKS charges ~$73/month for the control plane alone; GKE and AKS charge for worker nodes. Everything here is written for a **local cluster**, which is free:

```bash
brew install kind kubectl        # or: brew install minikube
kind create cluster --name campayn
```

## Deploy

The vector store is baked into the image, so build it first:

```bash
python scripts/ingest_knowledge_base.py --reset
```

```bash
# 1. build the image
docker build -t campayn-ai:latest .

# 2. make it visible to the cluster (kind has its own image store)
kind load docker-image campayn-ai:latest --name campayn
# minikube: minikube image load campayn-ai:latest

# 3. namespace + config
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-config.yaml

# 4. secrets — straight from .env, never from a committed file
kubectl create secret generic campayn-secret -n campayn --from-env-file=.env

# 5. data stores, then the app
kubectl apply -f k8s/03-postgres.yaml
kubectl apply -f k8s/04-redis.yaml
kubectl apply -f k8s/05-api.yaml
kubectl apply -f k8s/06-ui.yaml
```

## Check it

```bash
kubectl get pods -n campayn -w
```

Expect `postgres` and `redis` ready first — the API has an init container that blocks on `pg_isready`, so it stays in `Init:0/1` until Postgres accepts connections. First API start takes up to ~90s (loads 56k profiles, opens ChromaDB, compiles the graph); the startup probe allows 150s.

```bash
kubectl port-forward -n campayn svc/campayn-api 8000:8000
curl localhost:8000/health
```

Healthy output includes:

```json
{"postgres": "ok", "intent_cache": "ok (hits=0 misses=0 rate=0.0%)"}
```

UI: open **http://localhost:30085** (NodePort). With kind, map the port at cluster creation:

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30085
        hostPort: 30085
```
```bash
kind create cluster --name campayn --config kind-config.yaml
```

## Inspect the database

```bash
kubectl exec -it -n campayn postgres-0 -- psql -U campayn -d campayn

\dt                                    -- request_log, feedback, checkpoints…
SELECT count(*) FROM checkpoints;      -- written by PostgresSaver
SELECT * FROM feedback ORDER BY created_at DESC LIMIT 5;
```

## Why 2 API replicas is now safe

It wasn't before. `MemorySaver` kept checkpoints in process memory, so each replica had its own copy and a session's state was usually on the wrong pod. It also never evicted anything — roughly 115 KB per turn, ~7 checkpoints per request, cleared only by a restart. Moving checkpoints to Postgres makes the API pods stateless, which is what makes horizontal scaling and rolling updates work.

## Teardown

```bash
kubectl delete namespace campayn      # keeps the PVC's StorageClass data policy in mind
kind delete cluster --name campayn
```

## Known gaps

- **No Ingress/TLS.** NodePort only — fine locally, not for public traffic.
- **Postgres password is in a plain Secret.** Kubernetes Secrets are base64, not encrypted. Use sealed-secrets or a cloud secret manager for anything real.
- **Single Postgres replica**, no backups or PITR.
- **No HPA.** Would need metrics-server, which kind does not install by default.
- **Image carries the data.** Rebuilding the vector store means rebuilding the image; a PVC with an init job would be better at scale.

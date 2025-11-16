# Kubernetes Base Manifests

Базовые манифесты для Rap Analyzer в Kubernetes.

## Структура

```
k8s/base/
├── configmap.yaml       # Несекретные настройки
├── secret.yaml          # Секретные данные (API ключи, пароли)
├── deployment.yaml      # Описание Pod'ов
├── service.yaml         # Expose API внутри кластера
└── kustomization.yaml   # Kustomize конфиг
```

## Применение

```bash
# Просмотр итогового манифеста
kubectl kustomize k8s/base/

# Применить в кластер
kubectl apply -k k8s/base/

# Проверить
kubectl get pods -l app=rap-analyzer
kubectl logs -f deployment/rap-analyzer
```

## Важно! ⚠️

**Перед деплоем в production:**

1. Измени секреты в `secret.yaml`:
   ```bash
   # НЕ коммить в git реальные значения!
   kubectl create secret generic rap-analyzer-secrets \
     --from-literal=POSTGRES_PASSWORD='real_password' \
     --from-literal=NOVITA_API_KEY='real_key' \
     --dry-run=client -o yaml > secret.yaml
   ```

2. Настрой ресурсы в `deployment.yaml` под нагрузку

3. Добавь Ingress для внешнего доступа

## Best Practices

✅ **Сделано:**
- Non-root пользователь (UID 1000)
- Security context
- Health probes
- Resource limits
- Read-only filesystem (partially)

❌ **TODO:**
- Полностью read-only filesystem
- Network policies
- PodDisruptionBudget
- HorizontalPodAutoscaler

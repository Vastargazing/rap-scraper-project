{{/*
Expand the name of the chart.
*/}}
{{- define "rap-analyzer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "rap-analyzer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "rap-analyzer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "rap-analyzer.labels" -}}
helm.sh/chart: {{ include "rap-analyzer.chart" . }}
{{ include "rap-analyzer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "rap-analyzer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rap-analyzer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "rap-analyzer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "rap-analyzer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL labels
*/}}
{{- define "rap-analyzer.postgresql.labels" -}}
helm.sh/chart: {{ include "rap-analyzer.chart" . }}
app.kubernetes.io/name: {{ include "rap-analyzer.name" . }}-postgresql
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: database
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Monitoring labels
*/}}
{{- define "rap-analyzer.monitoring.labels" -}}
helm.sh/chart: {{ include "rap-analyzer.chart" . }}
app.kubernetes.io/name: {{ include "rap-analyzer.name" . }}-monitoring
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: monitoring
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Database connection string
*/}}
{{- define "rap-analyzer.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "rap-analyzer.name" . }}-postgresql:{{ .Values.postgresql.service.port }}/{{ .Values.postgresql.auth.database }}
{{- else }}
{{- required "A valid database URL must be provided when postgresql is disabled" .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "rap-analyzer.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- else if .Values.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}
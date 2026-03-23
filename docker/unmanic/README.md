# marcosviniciusi/unmanic:ocr

Imagem Docker customizada do Unmanic com suporte a OCR para conversao de legendas PGS/VOBSUB para SRT.

## O que inclui

Alem do Unmanic padrao (`josh5/unmanic:latest`):

| Pacote | Versao | Funcao |
|--------|--------|--------|
| tesseract-ocr | 5.x | Motor OCR para reconhecimento de texto em imagens |
| tesseract-ocr-por | best | Modelo PT-BR (tessdata_best — melhor precisao para acentos) |
| tesseract-ocr-eng | best | Modelo EN (tessdata_best) |
| mkvtoolnix | latest | Ferramentas para manipulacao de MKV (mkvextract) |
| pgsrip | latest | Biblioteca Python para OCR de legendas PGS |

## Build

```bash
cd docker/unmanic
docker build -t marcosviniciusi/unmanic:ocr .
```

## Run

### Docker Run

```bash
docker run -d \
  --name unmanic \
  -p 8888:8888 \
  -v /path/to/library:/library \
  -v /path/to/cache:/tmp/unmanic \
  -e PUID=1000 \
  -e PGID=1000 \
  marcosviniciusi/unmanic:ocr
```

### Docker Compose

```yaml
services:
  unmanic:
    image: marcosviniciusi/unmanic:ocr
    container_name: unmanic
    ports:
      - "8888:8888"
    volumes:
      - /path/to/library:/library
      - /path/to/cache:/tmp/unmanic
    environment:
      - PUID=1000
      - PGID=1000
    restart: unless-stopped
```

## Pipeline recomendada

Ordem dos plugins no Unmanic UI:

```
Filtros (on_library_management_file_test):
  1. ignore_files_recently_modified     ← pula arquivos sendo escritos
  2. vm_ignore_metadata_unmanic         ← pula arquivos ja processados
  3. vm_ignore_task_history             ← pula tarefas ja concluidas
  4. vm_ignore_video_over_res           ← pula videos acima do limite
  5. vm_ignore_video_under_res          ← pula videos abaixo do limite

Processamento (on_worker_process):
  6. vm_video_transcoder                ← transcodifica video (HW accel)
  7. vm_audio_transcoder                ← converte audio para EAC3 5.1
  8. vm_audio_transcode_create_stereo   ← cria downmix stereo
  9. vm_audio_remove_duplicates         ← remove audio duplicado
 10. vm_subtitles_pgs_to_srt           ← converte legendas para SRT (OCR)
 11. vm_subtitles_transcode             ← filtra legendas (mantem so PT-BR)
 12. vm_tag_pipeline_complete           ← marca arquivo como processado

Pos-processamento:
 13. vm_postprocessor_otel_trace        ← envia traces para SigNoz/Jaeger
```

## Verificacao

Dentro do container, verificar se tudo esta instalado:

```bash
docker exec unmanic tesseract --version
docker exec unmanic python3 -c "import pgsrip; print('pgsrip OK')"
docker exec unmanic mkvextract --version
docker exec unmanic tesseract --list-langs
```

## Troubleshooting

### tesseract nao encontra idioma portugues

```bash
docker exec unmanic tesseract --list-langs
# Deve listar: por
```

Se nao listar, verificar tessdata:
```bash
docker exec unmanic find / -name "por.traineddata" 2>/dev/null
```

### pgsrip nao importa

```bash
docker exec unmanic pip3 show pgsrip
```

### Precisao baixa nos acentos

O Dockerfile ja usa `tessdata_best` que tem a melhor precisao. Se ainda insuficiente,
considerar PaddleOCR como alternativa futura.

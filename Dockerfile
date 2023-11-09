FROM mambaorg/micromamba:0.15.3
USER root
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        apache2-utils \
        certbot \
        build-essential \
        python3-certbot-nginx \
        sudo \
        cifs-utils \
        && \
     rm -rf /var/lib/apt/lists/*

ENV HF_API_KEY="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"
ENV QD_URL="https://18919f34-5ce6-477a-bc2a-640a5a13027e.europe-west3-0.gcp.cloud.qdrant.io:6333"
ENV QD_API_KEY="3wK-BBWXxnyDK3EmHQqqJDEHA1I_kS4llvLm4MVaBpBjyZ9k8VFqHQ"
# RUN apt-get update && apt-get -y install cron
RUN mkdir /opt/streamlitapp
RUN chmod -R 777 /opt/streamlitapp
WORKDIR /opt/streamlitapp
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# RUN pip install sentence-transformers
COPY run.sh run.sh
COPY app.py app.py
COPY nginx.conf /etc/nginx/nginx.conf
USER root
# RUN touch /var/log/cron.log
# # Setup cron job
# RUN (crontab -l ; echo "0 12 * * * /usr/bin/certbot renew --quiet && tar -cpzf /mnt/letsencrypt/etc.tar.gz -C / etc/letsencrypt/ ") | crontab
RUN chmod a+x run.sh
CMD ["./run.sh"]

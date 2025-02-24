FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gpg-agent software-properties-common vim wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For libibex-dev
RUN add-apt-repository ppa:dreal/dreal --no-update -y

ENV DREAL_VERSION=4.21.06.2
ENV DREAL_DEBNAME=dreal_${DREAL_VERSION}_amd64.deb
ENV DREAL_URL=https://github.com/dreal/dreal4/releases/download/${DREAL_VERSION}/${DREAL_DEBNAME}
ENV DREAL_SHA256=c1798357bd967bf84b06fdaf7e962e102ff6703b3dee546fdc02862a1ecc09f1

RUN wget "${DREAL_URL}"

# Exit with error if SHA256 does not match
RUN if ! echo "${DREAL_SHA256}  ${DREAL_DEBNAME}" | sha256sum --check -; then \
        echo "SHA256 does not match ${DREAL_DEBNAME}:"; \
        echo "    expected: ${DREAL_SHA256}"; \
        echo "    actual  : $(sha256sum "${DREAL_DEBNAME}")"; \
        exit 1; \
    fi

# Install dReal from deb file
RUN apt-get update && apt-get install -y --no-install-recommends ./"${DREAL_DEBNAME}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    ; rm -f ./"${DREAL_DEBNAME}"

# Check dReal installation
RUN /opt/dreal/"${DREAL_VERSION}"/bin/dreal -v

# Install Python pip
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Creat and switch to a non-root user
RUN addgroup --system app && adduser --system --group app
USER app
WORKDIR /home/app

ENV PRICELY_REPO_URL=git@github.com:CyPhAi-Project/pricely.git#hscc2025
ADD --chown=app:app ${PRICELY_REPO_URL} ./pricely-repo
RUN python3 -m pip install --no-cache-dir --no-warn-script-location --upgrade pip && \
    python3 -m pip install --no-cache-dir --no-warn-script-location -r ./pricely-repo/docker/requirements.txt

CMD ["bash"]
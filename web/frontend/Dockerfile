FROM node:24-alpine AS build

WORKDIR /app

RUN corepack enable

COPY web/frontend/package.json web/frontend/pnpm-lock.yaml web/frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY web/frontend ./

ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN pnpm build

FROM nginx:1.29-alpine

COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

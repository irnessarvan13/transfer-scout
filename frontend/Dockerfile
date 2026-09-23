# start from official Node 20 image
FROM node:20-slim

# set working directory inside container
WORKDIR /app

# copy package files first — Docker caches this layer
COPY package*.json ./

# install all Node packages
RUN npm install

# copy the rest of the frontend code
COPY . .

# accept build arg and set as environment variable — baked into Vite bundle at build time
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# expose port 3000
EXPOSE 3000

# command to run when container starts
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
# Use Node 20
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm install

# Copy the rest of the application
COPY . .

# Expose port (default Vite dev port)
EXPOSE 5173

# Set environment variable for API URL (defaults to localhost for dev)
ENV VITE_API_BASE_URL=http://localhost:8000

# Start command (using --host to expose to Docker)
CMD ["npm", "run", "dev", "--", "--host"]

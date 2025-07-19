-- Initialize database for ML Explainer Dashboard
-- This script is run when the PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create users and databases if they don't exist
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ml_user') THEN
      CREATE ROLE ml_user LOGIN PASSWORD 'ml_password';
   END IF;
END
$$;

-- Grant necessary privileges
GRANT ALL PRIVILEGES ON DATABASE ml_explainer TO ml_user;
GRANT ALL ON SCHEMA public TO ml_user;
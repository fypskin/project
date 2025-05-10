import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = "https://xjngzorzjcrnkgiivuxy.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhqbmd6b3J6amNybmtnaWl2dXh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMDEyNzksImV4cCI6MjA1Nzc3NzI3OX0.cnD1ZDPUg-gUumAb-0D2Akl_uY7NZuFzIgKDyhkAyt4";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

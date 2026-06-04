const { createApp, ref, onMounted, nextTick, watch } = Vue;

createApp({
    setup() {
        // --- Authentication & View States ---
        const token = ref(localStorage.getItem("crm_token") || "");
        const currentView = ref("login");
        const settingsTab = ref("profile");
        const user = ref(null);
        const tenant = ref(null);
        const groqConfigured = ref(false);
        const loginForm = ref({ email: "", password: "" });
        const loginError = ref("");
        
        const registerForm = ref({
            signup_type: "create_tenant",
            tenant_name: "",
            tenant_id: "",
            name: "",
            email: "",
            password: ""
        });
        const registerError = ref("");
        const allTenants = ref([]);

        // --- Data Lists ---
        const clients = ref([]);
        const leads = ref([]);
        const opportunities = ref([]);
        const tenantUsers = ref([]);
        const chatMessages = ref([]);

        // --- Loading & UI States ---
        const pageLoading = ref(false);
        const chatLoading = ref(false);
        const chatInput = ref("");
        
        // --- Modals visibility ---
        const showClientModal = ref(false);
        const showLeadModal = ref(false);
        const showOppModal = ref(false);
        const showUserModal = ref(false);

        // --- CRUD Form bindings ---
        const clientForm = ref({ id: null, name: "", email: "", phone: "", company: "" });
        const leadForm = ref({ id: null, client_id: "", name_fallback: "", email_fallback: "", title: "", description: "", status: "New", value: 0 });
        const oppForm = ref({ id: null, lead_id: "", title: "", description: "", stage: "Discovery", value: 0, probability: 10, close_date: "" });
        const userForm = ref({ email: "", password: "", name: "", role: "user" });

        // --- Dashboard Stats ---
        const stats = ref({
            total_clients: 0,
            total_leads: 0,
            total_lead_value: 0,
            lead_status_distribution: {},
            total_opportunities: 0,
            pipeline_value: 0,
            opportunity_stage_distribution: {},
            recent_leads: [],
            recent_opportunities: []
        });

        // Charts references
        let leadChartInstance = null;
        let oppChartInstance = null;

        // --- Helper API request wrapper ---
        const apiRequest = async (url, options = {}) => {
            const headers = {
                "Content-Type": "application/json",
                ...options.headers
            };
            
            if (token.value) {
                headers["Authorization"] = `Bearer ${token.value}`;
            }

            const response = await fetch(url, { ...options, headers });
            
            if (response.status === 401) {
                // Token expired or invalid
                logout();
                throw new Error("Session expired. Please log in again.");
            }

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || "An error occurred during request.");
            }

            return response.json();
        };

        // --- Auth functions ---
        const handleLogin = async () => {
            loginError.value = "";
            pageLoading.value = true;
            try {
                const data = await apiRequest("/api/auth/login", {
                    method: "POST",
                    body: JSON.stringify(loginForm.value)
                });
                
                token.value = data.access_token;
                localStorage.setItem("crm_token", data.access_token);
                
                await loadUserProfile();
            } catch (err) {
                loginError.value = err.message;
            } finally {
                pageLoading.value = false;
            }
        };

        const handleRegister = async () => {
            registerError.value = "";
            pageLoading.value = true;
            try {
                const payload = { ...registerForm.value };
                if (payload.signup_type === "create_tenant") {
                    payload.tenant_id = null;
                } else {
                    payload.tenant_name = null;
                }

                const data = await apiRequest("/api/auth/register", {
                    method: "POST",
                    body: JSON.stringify(payload)
                });

                token.value = data.access_token;
                localStorage.setItem("crm_token", data.access_token);

                registerForm.value = {
                    signup_type: "create_tenant",
                    tenant_name: "",
                    tenant_id: "",
                    name: "",
                    email: "",
                    password: ""
                };

                await loadUserProfile();
            } catch (err) {
                registerError.value = err.message;
            } finally {
                pageLoading.value = false;
            }
        };

        const loadAllTenants = async () => {
            try {
                allTenants.value = await apiRequest("/api/auth/tenants");
            } catch (err) {
                console.error("Failed to load tenants list", err);
            }
        };

        const loadUserProfile = async () => {
            try {
                const me = await apiRequest("/api/auth/me");
                user.value = me;
                
                // Get Tenant and Groq Settings
                const config = await apiRequest("/api/settings");
                tenant.value = config.tenant;
                groqConfigured.value = config.groq_configured;

                // Route user based on role
                if (user.value.role === "admin") {
                    switchView("dashboard");
                } else {
                    switchView("chat");
                }
            } catch (err) {
                console.error("Failed to load profile", err);
                logout();
            }
        };

        const logout = () => {
            token.value = "";
            user.value = null;
            tenant.value = null;
            localStorage.removeItem("crm_token");
            currentView.value = "login";
            loginForm.value = { email: "", password: "" };
        };

        // --- Navigation ---
        const switchView = (view) => {
            currentView.value = view;
            
            // Fetch relevant view data
            if (view === "register") {
                loadAllTenants();
            } else if (view === "dashboard" && user.value?.role === "admin") {
                loadDashboardData();
            } else if (view === "clients" && user.value?.role === "admin") {
                loadClients();
            } else if (view === "leads" && user.value?.role === "admin") {
                loadLeads();
            } else if (view === "opportunities" && user.value?.role === "admin") {
                loadOpportunities();
            } else if (view === "chat" && user.value?.role === "user") {
                loadChatHistory();
            } else if (view === "settings") {
                loadSettingsData();
            }
        };

        // --- Dashboard Data & Charts ---
        const loadDashboardData = async () => {
            try {
                const data = await apiRequest("/api/dashboard");
                stats.value = data;
                
                // Render charts after DOM updates
                nextTick(() => {
                    renderCharts();
                });
            } catch (err) {
                alert("Error loading dashboard metrics: " + err.message);
            }
        };

        const renderCharts = () => {
            const leadCtx = document.getElementById("leadStatusChart");
            const oppCtx = document.getElementById("oppStageChart");

            if (!leadCtx || !oppCtx) return;

            // Destroy previous charts to avoid canvas overlaps
            if (leadChartInstance) leadChartInstance.destroy();
            if (oppChartInstance) oppChartInstance.destroy();

            // 1. Lead Doughnut Chart
            const leadData = stats.value.lead_status_distribution;
            const leadLabels = Object.keys(leadData);
            const leadValues = Object.values(leadData);

            leadChartInstance = new Chart(leadCtx, {
                type: "doughnut",
                data: {
                    labels: leadLabels.length ? leadLabels : ["No Leads"],
                    datasets: [{
                        data: leadValues.length ? leadValues : [0],
                        backgroundColor: ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981", "#ef4444"],
                        borderWidth: 1,
                        borderColor: "rgba(255, 255, 255, 0.08)"
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: "bottom",
                            labels: { color: "#9ca3af" }
                        }
                    }
                }
            });

            // 2. Opportunities Bar Chart
            const oppData = stats.value.opportunity_stage_distribution;
            const oppLabels = Object.keys(oppData);
            const oppValues = Object.values(oppData);

            oppChartInstance = new Chart(oppCtx, {
                type: "bar",
                data: {
                    labels: oppLabels.length ? oppLabels : ["No Opportunities"],
                    datasets: [{
                        label: "Value ($)",
                        data: oppValues.length ? oppValues : [0],
                        backgroundColor: "rgba(99, 102, 241, 0.4)",
                        borderColor: "#6366f1",
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            ticks: { color: "#9ca3af" },
                            grid: { color: "rgba(255, 255, 255, 0.05)" }
                        },
                        x: {
                            ticks: { color: "#9ca3af" },
                            grid: { display: false }
                        }
                    }
                }
            });
        };

        // --- Clients Actions ---
        const loadClients = async () => {
            try {
                clients.value = await apiRequest("/api/clients");
            } catch (err) {
                console.error("Error loading clients", err);
            }
        };

        const openAddClient = () => {
            clientForm.value = { id: null, name: "", email: "", phone: "", company: "" };
            showClientModal.value = true;
        };

        const openEditClient = (c) => {
            clientForm.value = { ...c };
            showClientModal.value = true;
        };

        const saveClient = async () => {
            try {
                if (clientForm.value.id) {
                    await apiRequest(`/api/clients/${clientForm.value.id}`, {
                        method: "PUT",
                        body: JSON.stringify(clientForm.value)
                    });
                } else {
                    await apiRequest("/api/clients", {
                        method: "POST",
                        body: JSON.stringify(clientForm.value)
                    });
                }
                showClientModal.value = false;
                loadClients();
            } catch (err) {
                alert("Error saving client: " + err.message);
            }
        };

        const deleteClient = async (id) => {
            if (!confirm("Are you sure you want to delete this client?")) return;
            try {
                await apiRequest(`/api/clients/${id}`, { method: "DELETE" });
                loadClients();
            } catch (err) {
                alert("Error deleting client: " + err.message);
            }
        };

        // --- Leads Actions ---
        const loadLeads = async () => {
            try {
                leads.value = await apiRequest("/api/leads");
                await loadClients(); // Needed for client list dropdown in form
            } catch (err) {
                console.error("Error loading leads", err);
            }
        };

        const openAddLead = () => {
            leadForm.value = { 
                id: null, 
                client_id: "", 
                name_fallback: "", 
                email_fallback: "", 
                title: "", 
                description: "", 
                status: "New", 
                value: 0 
            };
            showLeadModal.value = true;
        };

        const openEditLead = (l) => {
            leadForm.value = { 
                ...l, 
                client_id: l.client_id || "", 
                value: parseFloat(l.value) 
            };
            showLeadModal.value = true;
        };

        const saveLead = async () => {
            try {
                // Formatting payload
                const payload = { ...leadForm.value };
                if (payload.client_id === "") payload.client_id = null;
                if (!payload.client_id) {
                    if (!payload.name_fallback) {
                        alert("Please specify a client name or select an existing client.");
                        return;
                    }
                } else {
                    payload.name_fallback = null;
                    payload.email_fallback = null;
                }

                if (payload.id) {
                    await apiRequest(`/api/leads/${payload.id}`, {
                        method: "PUT",
                        body: JSON.stringify(payload)
                    });
                } else {
                    await apiRequest("/api/leads", {
                        method: "POST",
                        body: JSON.stringify(payload)
                    });
                }
                showLeadModal.value = false;
                loadLeads();
            } catch (err) {
                alert("Error saving lead: " + err.message);
            }
        };

        const deleteLead = async (id) => {
            if (!confirm("Are you sure you want to delete this lead?")) return;
            try {
                await apiRequest(`/api/leads/${id}`, { method: "DELETE" });
                loadLeads();
            } catch (err) {
                alert("Error deleting lead: " + err.message);
            }
        };

        // --- Opportunities Actions ---
        const loadOpportunities = async () => {
            try {
                opportunities.value = await apiRequest("/api/opportunities");
                await loadLeads(); // Needed for lead list dropdown in form
            } catch (err) {
                console.error("Error loading opportunities", err);
            }
        };

        const openAddOpp = () => {
            oppForm.value = { 
                id: null, 
                lead_id: "", 
                title: "", 
                description: "", 
                stage: "Discovery", 
                value: 0, 
                probability: 10, 
                close_date: "" 
            };
            showOppModal.value = true;
        };

        const openEditOpp = (o) => {
            oppForm.value = { 
                ...o, 
                lead_id: o.lead_id || "", 
                value: parseFloat(o.value),
                close_date: o.close_date || ""
            };
            showOppModal.value = true;
        };

        const saveOpp = async () => {
            try {
                const payload = { ...oppForm.value };
                if (payload.lead_id === "") payload.lead_id = null;
                if (payload.close_date === "") payload.close_date = null;

                if (payload.id) {
                    await apiRequest(`/api/opportunities/${payload.id}`, {
                        method: "PUT",
                        body: JSON.stringify(payload)
                    });
                } else {
                    await apiRequest("/api/opportunities", {
                        method: "POST",
                        body: JSON.stringify(payload)
                    });
                }
                showOppModal.value = false;
                loadOpportunities();
            } catch (err) {
                alert("Error saving opportunity: " + err.message);
            }
        };

        const deleteOpp = async (id) => {
            if (!confirm("Are you sure you want to delete this opportunity?")) return;
            try {
                await apiRequest(`/api/opportunities/${id}`, { method: "DELETE" });
                loadOpportunities();
            } catch (err) {
                alert("Error deleting opportunity: " + err.message);
            }
        };

        // --- Chat Operations ---
        const loadChatHistory = async () => {
            try {
                const history = await apiRequest("/api/chat/history");
                chatMessages.value = history.map(msg => ({
                    ...msg,
                    showSql: false
                }));
                scrollChatToBottom();
            } catch (err) {
                console.error("Error loading chat history", err);
            }
        };

        const sendChatMessage = async () => {
            const prompt = chatInput.value.trim();
            if (!prompt) return;

            chatInput.value = "";
            chatMessages.value.push({
                role: "user",
                content: prompt,
                created_at: new Date().toISOString()
            });

            scrollChatToBottom();
            chatLoading.value = true;

            try {
                const reply = await apiRequest("/api/chat", {
                    method: "POST",
                    body: JSON.stringify({ content: prompt })
                });

                chatMessages.value.push({
                    role: "assistant",
                    content: reply.response,
                    sql_executed: reply.sql_executed,
                    raw_results: reply.raw_results,
                    showSql: false,
                    created_at: new Date().toISOString()
                });
            } catch (err) {
                chatMessages.value.push({
                    role: "assistant",
                    content: "Sorry, I ran into an error processing that question: " + err.message,
                    created_at: new Date().toISOString()
                });
            } finally {
                chatLoading.value = false;
                scrollChatToBottom();
            }
        };

        const toggleSql = (msg) => {
            msg.showSql = !msg.showSql;
        };

        const scrollChatToBottom = () => {
            nextTick(() => {
                const container = document.getElementById("chatHistoryContainer");
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        };

        // --- Settings Management ---
        const loadSettingsData = async () => {
            if (user.value.role === "admin") {
                loadTenantUsers();
            }
        };

        const loadTenantUsers = async () => {
            try {
                tenantUsers.value = await apiRequest("/api/settings/users");
            } catch (err) {
                console.error("Failed to load users list", err);
            }
        };

        const openAddUser = () => {
            userForm.value = { email: "", password: "", name: "", role: "user" };
            showUserModal.value = true;
        };

        const saveUser = async () => {
            try {
                await apiRequest("/api/settings/users", {
                    method: "POST",
                    body: JSON.stringify(userForm.value)
                });
                showUserModal.value = false;
                loadTenantUsers();
            } catch (err) {
                alert("Error creating user: " + err.message);
            }
        };

        // --- Formatting utils ---
        const formatCurrency = (val) => {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
        };

        const formatDate = (dateStr) => {
            if (!dateStr) return "N/A";
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        };

        // --- Lifecycle ---
        onMounted(() => {
            if (token.value) {
                loadUserProfile();
            } else {
                currentView.value = "login";
            }
        });

        return {
            // States
            token,
            currentView,
            settingsTab,
            user,
            tenant,
            groqConfigured,
            loginForm,
            loginError,
            registerForm,
            registerError,
            allTenants,
            pageLoading,
            chatLoading,
            chatInput,
            chatMessages,
            stats,
            
            // CRUD collections
            clients,
            leads,
            opportunities,
            tenantUsers,

            // Forms & Modals
            showClientModal,
            showLeadModal,
            showOppModal,
            showUserModal,
            clientForm,
            leadForm,
            oppForm,
            userForm,

            // Actions
            handleLogin,
            handleRegister,
            loadAllTenants,
            logout,
            switchView,
            
            // Clients operations
            openAddClient,
            openEditClient,
            saveClient,
            deleteClient,

            // Leads operations
            openAddLead,
            openEditLead,
            saveLead,
            deleteLead,

            // Opportunities operations
            openAddOpp,
            openEditOpp,
            saveOpp,
            deleteOpp,

            // Chat operations
            sendChatMessage,
            toggleSql,
            
            // Settings operations
            openAddUser,
            saveUser,

            // Formatting
            formatCurrency,
            formatDate
        };
    }
}).mount("#app");

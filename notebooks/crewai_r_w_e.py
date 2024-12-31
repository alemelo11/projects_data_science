
# 🤖 Agentes Múltiplos - crewai 🤖

from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Agora, você pode acessar as variáveis de ambiente com os.getenv
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Ferramentas para o seu processo
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# Definindo seus agentes com papéis, objetivos e histórias de fundo específicas
content_researcher = Agent(
    role="Pesquisador de Conteúdo",
    goal="Busque conteúdo on line sobre o tema {tema} ",
    backstory="Um pesquisador e escritor proficiente com habilidade para escrever artigos no Medium e LinkedIn sobre {tema}."
              "Você fará uma busca sobre informações na Internet, coletando e agrupando-as."
              "Seu trabalho servirá como base para o content_writer.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)

content_writer = Agent(
    role="Redator de Conteúdo",
    goal="Escrever um texto envolvente e informativo para o Medium e outro para o LinkedIn sobre {tema}",
    backstory="Um redator experiente que pode criar textos atraentes, interessantes e divertidos para várias"
              "plataformas (especialmente Medium e Linkedin), especializado em tópicos sobre {tema}."
              "Dê opiniões sobre o tema, mas deixe claro que são opiniões pessoais."
              "O artigo do Medium pode ser mais abrangente ou seja, maior em tamanho; e o artigo"
              "do Linkedin pode ser um resumo do primeiro",
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)


content_editor = Agent(
    role="Editor de Conteúdo",
    goal="Garantir que o texto seja polido, gramaticalmente correto e adequado ao público do Medium e LinkedIn sobre {tema}.",
    backstory="Um editor meticuloso com olhos para os detalhes, responsável por receber textos do content_writer,"
              "refinar e aprimorar o conteúdo para impacto máximo.",
    verbose=True,
    allow_delegation=False,
)

# Definindo seus agentes com papéis, objetivos e histórias de fundo específicas

content_researcher = Agent(
    role="Pesquisador de Conteúdo",
    goal="Buscar conteúdo on-line sobre o tema {tema}.",
    backstory=(
        "Um pesquisador e escritor proficiente com habilidade para criar artigos no Medium e LinkedIn sobre {tema}. "
        "Você realizará buscas na Internet, coletando e agrupando informações. "
        "Seu trabalho servirá como base para o content_writer."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)

content_writer = Agent(
    role="Redator de Conteúdo",
    goal="Escrever um texto envolvente e informativo para o Medium  e outro para o Linkedin sobre {tema}.",
    backstory=(
        "Um redator experiente que pode criar textos atraentes, interessantes e dinâmicos para várias "
        "plataformas (especialmente Medium e LinkedIn), especializado em tópicos sobre {tema}. "
        "Dê opiniões sobre o tema, mas deixe claro que são opiniões pessoais. "
        "O artigo do Medium pode ser mais abrangente (maior em tamanho), enquanto o artigo do LinkedIn deve ser um resumo do primeiro."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)

content_editor = Agent(
    role="Editor de Conteúdo",
    goal="Garantir que o texto seja polido, gramaticalmente correto e adequado ao público do Medium e LinkedIn sobre {tema}.",
    backstory=(
        "Um editor meticuloso, com olhos para detalhes, responsável por receber textos do content_writer "
        "e refiná-los para maximizar o impacto e adequação ao público-alvo."
    ),
    verbose=True,
    allow_delegation=False,
)




# Criando tarefas para os agentes

task_research = Task(
    description=(
        "Realizar uma pesquisa abrangente sobre {tema}. "
        "Identificar os princípios-chave, vantagens e exemplos do mundo real de sua aplicação. "
        "Reunir informações suficientes para apoiar a criação de um texto informativo e envolvente para o Medium e outro para o LinkedIn."
        "O artigo do Medium pode ser mais abrangente (maior em tamanho), enquanto o artigo do LinkedIn deve ser um resumo do primeiro."
    ),
    agent=content_researcher,
    expected_output=(
        "Um conjunto detalhado de informações organizadas sobre {tema}, incluindo seus princípios-chave, "
        "vantagens, exemplos do mundo real e referências confiáveis para o content_writer utilizar."
    ),
)

task_write = Task(
    description=(
        "Com base nos insights reunidos pelo Pesquisador de Conteúdo, crie um texto para o Medium e outro para o LinkedIn que explique sobre {tema}. "
        "O texto deve incluir:\n"
        "- Uma introdução cativante que prenda a atenção dos profissionais no Medium e LinkedIn.\n"
        "- Uma explicação clara sobre {tema} e seus princípios fundamentais.\n"
        "- Os benefícios de {tema} em sua área de aplicação.\n"
        "- Exemplos breves de como {tema} é usado em cenários do mundo real.\n"
        "- Uma conclusão que resuma os pontos-chave e incentive o engajamento dos leitores.\n"
        "- O texto deve ser conciso, informativo e adaptado ao público profissional do LinkedIn.\n"
        "- Use hashtags relevantes para aumentar a descoberta."
        "O artigo do Medium pode ser mais abrangente (maior em tamanho), enquanto o artigo do LinkedIn deve ser um resumo do primeiro."
    ),
    agent=content_writer,
    expected_output=(
        "Um texto conciso e informativo para Medium e outro para  LinkedIn sobre {tema}, com aproximadamente 500 a 800 palavras, "
        "estruturado para engajar o público profissional e com hashtags relevantes para aumentar a descoberta."
        "O artigo do Medium pode ser mais abrangente (maior em tamanho), enquanto o artigo do LinkedIn deve ser um resumo do primeiro."
    ),
)

task_edit = Task(
    description=(
        "Revisar o texto do LinkedIn sobre {tema} gerado pelo Redator de Conteúdo. "
        "Concentre-se nos seguintes aspectos:\n"
        "- Certifique-se de que o texto esteja bem estruturado, gramaticalmente correto e livre de erros.\n"
        "- Refinar a linguagem para clareza, concisão e engajamento.\n"
        "- Verificar a adequação do texto ao público do Medium e do LinkedIn, garantindo que ele mantenha um tom profissional e perspicaz.\n"
        "- Fazer os ajustes necessários para melhorar a legibilidade e o impacto geral.\n"
        "- O comprimento ideal do texto deve estar entre 500 e 800 palavras."
        "-O artigo do Medium pode ser mais abrangente (maior em tamanho), enquanto o artigo do LinkedIn deve ser um resumo do primeiro."
    ),
    agent=content_editor,
    expected_output=(
        "Uma versão final polida e gramaticalmente correta do texto sobre {tema}, adaptada para o público do Medium e LinkedIn, "
        "pronta para publicação com impacto máximo."
    ),
)


equipe = Crew(
    agents=[content_researcher, content_writer, content_editor],
    tasks=[task_research, task_write, task_edit],
    verbose=True
)

# Rodando o crew
tema_do_artigo = 'Agentes de Inteligência Artificial'
input_data = {'tema': tema_do_artigo}
result = equipe.kickoff(inputs=input_data)

from IPython.display import Markdown
Markdown(result)
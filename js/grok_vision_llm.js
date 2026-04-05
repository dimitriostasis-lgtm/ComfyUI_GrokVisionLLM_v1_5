import { app } from "/scripts/app.js";

app.registerExtension({
  name: "ComfyUI.GrokVisionLLM.MaskApiKey",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "GrokVisionLLM") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      const applyMask = () => {
        const apiKeyWidget = this.widgets?.find((w) => w.name === "api_key");
        if (apiKeyWidget?.inputEl) {
          apiKeyWidget.inputEl.type = "password";
          apiKeyWidget.inputEl.autocomplete = "off";
          apiKeyWidget.inputEl.spellcheck = false;
        }
      };

      applyMask();
      setTimeout(applyMask, 0);
      setTimeout(applyMask, 100);

      return result;
    };
  },
});
